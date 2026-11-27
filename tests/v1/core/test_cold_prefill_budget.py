# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from types import SimpleNamespace

import pytest

import vllm.envs as envs
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler, mock_kv


def budget(monkeypatch, limit=1, **kwargs):
    monkeypatch.setattr(envs, "VLLM_MAX_COLD_PREFILLS", limit)
    return create_scheduler(
        max_num_batched_tokens=32,
        long_prefill_token_threshold=8,
        max_model_len=256,
        skip_tokenizer_init=True,
        **kwargs,
    )


def request(scheduler, name, length):
    (req,) = create_requests(1, num_tokens=length, req_ids=[name])
    scheduler.add_request(req)
    return req


def complete(scheduler, step, sampled=()):
    ids = list(step.num_scheduled_tokens)
    scheduler.update_from_output(
        step,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={rid: i for i, rid in enumerate(ids)},
            sampled_token_ids=[list(sampled) for _ in ids],
        ),
    )


@pytest.mark.parametrize("lanes", [1, 4])
def test_cancellation_releases_only_owned_slot(monkeypatch, lanes):
    scheduler = budget(monkeypatch, max_parallel_prefills=lanes)
    owner = request(scheduler, "owner", 96)
    complete(scheduler, scheduler.schedule())
    blocked = request(scheduler, "blocked", 32)
    cheap = request(scheduler, "cheap", 31)
    step = scheduler.schedule()
    assert cheap.request_id in step.num_scheduled_tokens
    assert blocked.request_id not in step.num_scheduled_tokens
    assert scheduler._cold_prefill_ends.keys() == {owner.request_id}
    complete(scheduler, step)
    scheduler.finish_requests(blocked.request_id, RequestStatus.FINISHED_ABORTED)
    assert scheduler._cold_prefill_ends.keys() == {owner.request_id}
    scheduler.finish_requests(owner.request_id, RequestStatus.FINISHED_ABORTED)
    assert not scheduler._cold_prefill_ends
    next_owner = request(scheduler, "next", 32)
    assert next_owner.request_id in scheduler.schedule().num_scheduled_tokens


def test_recheck_reclassified_waiter(monkeypatch):
    scheduler = budget(monkeypatch, enable_prefix_caching=True)
    owner = request(scheduler, "shared-producer", 128)
    complete(scheduler, scheduler.schedule())
    follower = request(scheduler, "shared-follower", 65)
    for i in range(14):
        step = scheduler.schedule()
        if follower.request_id in step.num_scheduled_tokens:
            assert i > 0
            assert (
                follower.num_computed_tokens
                - step.num_scheduled_tokens[follower.request_id]
                >= 48
            )
            assert scheduler._cold_prefill_ends.keys() == {owner.request_id}
            break
        complete(scheduler, step)
    else:
        pytest.fail("A newly reusable prefix did not bypass the occupied slot")


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_release_on_worker_completion(monkeypatch, async_scheduling):
    scheduler = budget(monkeypatch, async_scheduling=async_scheduling)
    owner = request(scheduler, "owner", 32)
    waiter = request(scheduler, "waiter", 32)
    pending = []
    for _ in range(3):
        step = scheduler.schedule()
        assert waiter.request_id not in step.num_scheduled_tokens
        if async_scheduling:
            pending.append(step)
        else:
            complete(scheduler, step)
    final = scheduler.schedule()
    assert waiter.request_id not in final.num_scheduled_tokens
    assert owner.num_computed_tokens == owner.num_prompt_tokens
    assert scheduler._cold_prefill_ends.keys() == {"owner"}
    for step in pending:
        complete(scheduler, step)
        assert scheduler._cold_prefill_ends.keys() == {"owner"}
    complete(scheduler, final)
    assert not scheduler._cold_prefill_ends
    assert waiter.request_id in scheduler.schedule().num_scheduled_tokens
    assert scheduler._cold_prefill_ends.keys() == {waiter.request_id}


@pytest.mark.parametrize("fair", [False, True])
def test_preemption_releases_slot(monkeypatch, fair):
    monkeypatch.setattr(envs, "VLLM_FAIR_COLD_PREFILLS", fair)
    monkeypatch.setattr(envs, "VLLM_COLD_PREFILL_CALLER_LIMIT", 1)
    scheduler = budget(monkeypatch)
    owner = request(scheduler, "owner", 64)
    complete(scheduler, scheduler.schedule())
    scheduler.running.remove(owner)
    scheduler._preempt_request(owner, time.monotonic())
    assert not scheduler._cold_prefill_ends
    assert owner.num_preemptions == 1
    assert owner.request_id in scheduler.schedule().num_scheduled_tokens
    assert scheduler._cold_prefill_ends.keys() == {owner.request_id}


@pytest.mark.parametrize(
    "limit,kwargs",
    [
        (0, {}),
        (1, {"scheduling_policy": "priority"}),
        (1, {"use_kv_connector": mock_kv(0, False)}),
        (1, {"use_ec_connector": True, "ec_role": "ec_consumer"}),
    ],
)
@pytest.mark.parametrize("fair", [False, True])
def test_unrestricted_paths(monkeypatch, limit, kwargs, fair):
    monkeypatch.setattr(envs, "VLLM_FAIR_COLD_PREFILLS", fair)
    scheduler = budget(monkeypatch, limit, **kwargs)
    assert scheduler._cold_caller_limit == 0
    request(scheduler, "one", 64)
    request(scheduler, "two", 64)
    assert set(scheduler.schedule().num_scheduled_tokens) == {"one", "two"}
    assert not scheduler._cold_prefill_ends


def test_failed_allocation_does_not_charge_budget(monkeypatch):
    scheduler = budget(monkeypatch)
    request(scheduler, "cannot-allocate", 32)
    monkeypatch.setattr(
        scheduler.kv_cache_manager, "allocate_slots", lambda *a, **k: None
    )
    assert not scheduler.schedule().num_scheduled_tokens
    assert not scheduler._cold_prefill_ends


def test_negative_limit_rejected(monkeypatch):
    with pytest.raises(ValueError, match="nonnegative"):
        budget(monkeypatch, -1)


def test_eight_slots_acquired_in_one_schedule(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MAX_COLD_PREFILLS", 8)
    scheduler = create_scheduler(
        max_num_batched_tokens=128,
        long_prefill_token_threshold=8,
        max_model_len=256,
        skip_tokenizer_init=True,
    )
    for req in create_requests(12, num_tokens=129):
        scheduler.add_request(req)
    step = scheduler.schedule()
    assert len(step.num_scheduled_tokens) == 8
    assert scheduler._cold_prefill_ends.keys() == set(step.num_scheduled_tokens)


def test_parked_requests_keep_fcfs_order(monkeypatch):
    scheduler = budget(monkeypatch)
    owner = request(scheduler, "owner", 96)
    complete(scheduler, scheduler.schedule())
    for name in ("older", "younger"):
        request(scheduler, name, 64)
    for _ in range(2):
        complete(scheduler, scheduler.schedule())
        assert [r.request_id for r in scheduler.skipped_waiting] == ["older", "younger"]
    scheduler.finish_requests(owner.request_id, RequestStatus.FINISHED_ABORTED)
    assert set(scheduler.schedule().num_scheduled_tokens) == {"older"}


@pytest.mark.parametrize("fair", [False, True])
def test_large_streaming_update_waits_for_slot(monkeypatch, fair):
    monkeypatch.setattr(envs, "VLLM_FAIR_COLD_PREFILLS", fair)
    monkeypatch.setattr(envs, "VLLM_COLD_PREFILL_CALLER_LIMIT", 1)
    scheduler = budget(monkeypatch)
    (session,) = create_requests(1, num_tokens=8, max_tokens=1, req_ids=["stream"])
    session.resumable = True
    scheduler.add_request(session)
    complete(scheduler, scheduler.schedule(), [17])
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ

    owner = request(scheduler, "owner", 96)
    complete(scheduler, scheduler.schedule())
    (update,) = create_requests(1, num_tokens=32, req_ids=["stream"])
    update.resumable = True
    scheduler.add_request(update)
    step = scheduler.schedule()
    assert session.request_id not in step.num_scheduled_tokens
    complete(scheduler, step)
    scheduler.finish_requests(owner.request_id, RequestStatus.FINISHED_ABORTED)
    assert session.request_id in scheduler.schedule().num_scheduled_tokens
    assert scheduler._cold_prefill_ends == {"stream": 40}


def test_recompute_keeps_slot_through_generated_history(monkeypatch):
    scheduler = budget(monkeypatch)
    (session,) = create_requests(1, num_tokens=8, max_tokens=64, req_ids=["recompute"])
    scheduler.add_request(session)
    for _ in range(24):
        complete(scheduler, scheduler.schedule(), [17])
    assert session.num_tokens == 32
    scheduler.running.remove(session)
    scheduler._preempt_request(session, time.monotonic())
    for _ in range(3):
        complete(scheduler, scheduler.schedule())
        assert scheduler._cold_prefill_ends == {"recompute": 32}
    complete(scheduler, scheduler.schedule())
    assert not scheduler._cold_prefill_ends


@pytest.mark.parametrize("drop_stale", [False, True])
def test_stale_output_and_abort_all(monkeypatch, drop_stale):
    scheduler = budget(monkeypatch, async_scheduling=True)
    owner = request(scheduler, "owner", 64)
    stale = scheduler.schedule()
    scheduler.running.remove(owner)
    scheduler._preempt_request(owner, time.monotonic(), drop_stale_output=drop_stale)
    current = scheduler.schedule()
    if drop_stale:
        assert owner.request_id in current.num_scheduled_tokens
        complete(scheduler, stale)
    else:
        assert not current.num_scheduled_tokens
        complete(scheduler, stale)
        assert not scheduler._cold_prefill_ends
        current = scheduler.schedule()
        assert owner.request_id in current.num_scheduled_tokens
    assert scheduler._cold_prefill_ends == {"owner": 64}
    request(scheduler, "waiter", 64)
    scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
    complete(scheduler, current)
    assert not scheduler._cold_prefill_ends


def caller_budget(monkeypatch, *, fair=True, share=2, **kwargs):
    monkeypatch.setattr(envs, "VLLM_MAX_COLD_PREFILLS", 8)
    monkeypatch.setattr(envs, "VLLM_FAIR_COLD_PREFILLS", fair)
    monkeypatch.setattr(envs, "VLLM_COLD_PREFILL_CALLER_LIMIT", share)
    return create_scheduler(
        max_num_batched_tokens=128,
        long_prefill_token_threshold=8,
        max_model_len=256,
        skip_tokenizer_init=True,
        **kwargs,
    )


def caller_request(scheduler, name, caller, length=129):
    (req,) = create_requests(1, num_tokens=length, req_ids=[name])
    if caller is not None:
        req.sampling_params.extra_args = {"cold_prefill_caller": caller}
    scheduler.add_request(req)
    return req


@pytest.mark.parametrize("caller", ["bulk", None])
def test_one_caller_borrows_all_eight(monkeypatch, caller):
    scheduler = caller_budget(monkeypatch)
    for i in range(10):
        caller_request(scheduler, str(i), caller)
    assert list(scheduler.schedule().num_scheduled_tokens) == [str(i) for i in range(8)]
    assert len(scheduler._cold_prefill_ends) == 8


def test_invalid_caller_values_share_anonymous_budget(monkeypatch):
    scheduler = caller_budget(monkeypatch)
    for i, label in enumerate([None] * 8 + ["", "   ", 42, [], {}, "real"]):
        caller_request(scheduler, str(i), label)
    step = scheduler.schedule()
    assert list(step.num_scheduled_tokens) == ["0", "1", "13", "2", "3", "4", "5", "6"]


@pytest.mark.parametrize("fair", [True, False])
def test_fair_share_preference_fallback_and_caller_order(monkeypatch, fair):
    scheduler = caller_budget(monkeypatch, fair=fair)
    for label, count in [("a", 9), ("b", 3), ("c", 2)]:
        for i in range(count):
            caller_request(scheduler, f"{label}{i}", label)
    step = scheduler.schedule()
    expected = ["a0", "a1", "b0", "b1", "c0", "c1", "a2", "a3"]
    assert list(step.num_scheduled_tokens) == (
        expected if fair else [f"a{i}" for i in range(8)]
    )


@pytest.mark.parametrize("lanes", [1, 4])
def test_late_caller_gets_a_new_slot_without_revoking_owners(monkeypatch, lanes):
    scheduler = caller_budget(monkeypatch, max_parallel_prefills=lanes)
    for i in range(10):
        caller_request(scheduler, f"a{i}", "a")
    for _ in range(8):
        complete(scheduler, scheduler.schedule())
        if len(scheduler._cold_prefill_ends) == 8:
            break
    owners = set(scheduler._cold_prefill_ends)
    assert len(owners) == 8
    caller_request(scheduler, "cancelled", "b")
    caller_request(scheduler, "late", "c")
    scheduler.finish_requests("cancelled", RequestStatus.FINISHED_ABORTED)
    complete(scheduler, scheduler.schedule())
    assert scheduler._cold_prefill_ends.keys() == owners
    scheduler.finish_requests("a0", RequestStatus.FINISHED_ABORTED)
    for _ in range(4):
        step = scheduler.schedule()
        if "late" in step.num_scheduled_tokens:
            break
        complete(scheduler, step)
    assert scheduler._cold_prefill_ends.keys() == (owners - {"a0"}) | {"late"}


def test_async_loan_passes_to_under_share_caller_after_worker_completion(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_FAIR_COLD_PREFILLS", True)
    monkeypatch.setattr(envs, "VLLM_COLD_PREFILL_CALLER_LIMIT", 1)
    scheduler = budget(monkeypatch, 2, async_scheduling=True)
    caller_request(scheduler, "owner", "bulk", 96)
    loan = caller_request(scheduler, "loan", "bulk", 32)
    for _ in range(2):
        complete(scheduler, scheduler.schedule())
    assert scheduler._cold_prefill_ends == {"owner": 96, "loan": 32}

    caller_request(scheduler, "older", "bulk", 64)
    caller_request(scheduler, "late", "occasional", 64)
    penultimate = scheduler.schedule()
    final = scheduler.schedule()
    assert penultimate.num_scheduled_tokens["loan"] == 8
    assert final.num_scheduled_tokens["loan"] == 8
    assert loan.num_in_flight_tokens == 16
    complete(scheduler, penultimate)
    assert loan.num_in_flight_tokens == 8
    waiting_step = scheduler.schedule()
    assert not {"older", "late"}.intersection(waiting_step.num_scheduled_tokens)
    assert scheduler._cold_prefill_ends == {"owner": 96, "loan": 32}

    complete(scheduler, final)
    assert (
        loan.num_computed_tokens - loan.num_in_flight_tokens >= loan.num_prompt_tokens
    )
    assert scheduler._cold_prefill_ends.keys() == {"owner"}
    next_step = scheduler.schedule()
    assert {"older", "late"}.intersection(next_step.num_scheduled_tokens) == {"late"}
    assert scheduler._cold_prefill_ends == {"owner": 96, "late": 64}
    assert "older" in {
        r.request_id for r in (*scheduler.waiting, *scheduler.skipped_waiting)
    }


@pytest.mark.parametrize("lanes", [1, "auto"])
def test_failed_competitor_allocation_cannot_strand_a_loan(monkeypatch, lanes):
    scheduler = caller_budget(monkeypatch, max_parallel_prefills=lanes)
    for i in range(9):
        caller_request(scheduler, f"a{i}", "a")
    caller_request(scheduler, "cannot-fit", "b")
    caller_request(scheduler, "fits", "c")
    allocate = scheduler.kv_cache_manager.allocate_slots
    attempts = []

    def allocate_fitting(req, *args, **kwargs):
        attempts.append(req.request_id)
        return (
            None if req.request_id == "cannot-fit" else allocate(req, *args, **kwargs)
        )

    monkeypatch.setattr(scheduler.kv_cache_manager, "allocate_slots", allocate_fitting)
    scheduled = list(scheduler.schedule().num_scheduled_tokens)
    assert attempts.index("cannot-fit") < attempts.index("fits")
    assert scheduled == (
        [
            "a0",
            "a1",
            "fits",
            "a2",
            "a3",
            "a4",
            "a5",
            "a6",
        ]
        if lanes == 1
        else ["a0", "a1", "fits", "a3"]
    )


@pytest.mark.parametrize("fair", [False, True])
def test_failed_fcfs_head_still_waits_for_capacity(monkeypatch, fair):
    scheduler = caller_budget(monkeypatch, fair=fair, share=1)
    caller_request(scheduler, "owner", "bulk")
    complete(scheduler, scheduler.schedule())
    head = caller_request(scheduler, "head", "occasional")
    caller_request(scheduler, "borrower", "bulk")
    allocate = scheduler.kv_cache_manager.allocate_slots
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "allocate_slots",
        lambda req, *a, **kw: None if req is head else allocate(req, *a, **kw),
    )
    complete(scheduler, scheduler.schedule())
    assert scheduler._cold_prefill_ends.keys() == {"owner"}
    scheduler.finish_requests("head", RequestStatus.FINISHED_ABORTED)
    assert "borrower" in scheduler.schedule().num_scheduled_tokens


def test_oversized_unchunked_competitor_leaves_room_for_borrower(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MAX_COLD_PREFILLS", 2)
    monkeypatch.setattr(envs, "VLLM_FAIR_COLD_PREFILLS", True)
    monkeypatch.setattr(envs, "VLLM_COLD_PREFILL_CALLER_LIMIT", 1)
    monkeypatch.setattr(envs, "VLLM_SCHEDULER_UNCAP_PREFILL_ONLY_STEPS", True)
    scheduler = create_scheduler(
        max_num_batched_tokens=64,
        max_model_len=64,
        enable_chunked_prefill=False,
        skip_tokenizer_init=True,
    )
    # All three prompts are costly at 16 tokens. The prefill-only batch has
    # 32 tokens left after the owner, enough for the borrower but not its rival.
    scheduler.max_num_scheduled_tokens = 16
    caller_request(scheduler, "owner", "bulk", 32)
    caller_request(scheduler, "borrower", "bulk", 24)
    caller_request(scheduler, "oversized", "occasional", 48)
    assert scheduler.schedule(throttle_prefills=True).num_scheduled_tokens == {
        "owner": 32,
        "borrower": 24,
    }


@pytest.mark.parametrize("ready", [False, True])
def test_fair_grammar_waiter_can_use_an_existing_lane_when_ready(monkeypatch, ready):
    scheduler = caller_budget(monkeypatch, max_parallel_prefills="auto")
    # Parked bulk requests precede a grammar whose readiness is checked on admission.
    for i in range(6):
        req = caller_request(scheduler, f"a{i}", "bulk")
        scheduler.waiting.remove_request(req)
        scheduler.skipped_waiting.add_request(req)
    grammar = caller_request(scheduler, "grammar", "occasional")
    scheduler.waiting.remove_request(grammar)
    grammar.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    grammar.structured_output_request = SimpleNamespace(
        grammar=object() if ready else None
    )
    scheduler.skipped_waiting.add_request(grammar)
    scheduled = list(scheduler.schedule().num_scheduled_tokens)
    assert scheduled == ["a0", "a1", "grammar" if ready else "a2", "a3"]
    if not ready:
        assert grammar.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR


@pytest.mark.parametrize("competing", [False, True])
def test_promoted_over_share_grammar_yields_only_to_a_competitor(
    monkeypatch, competing
):
    monkeypatch.setattr(envs, "VLLM_FAIR_COLD_PREFILLS", True)
    monkeypatch.setattr(envs, "VLLM_COLD_PREFILL_CALLER_LIMIT", 1)
    scheduler = budget(monkeypatch, 2, max_parallel_prefills="auto")
    caller_request(scheduler, "owner", "bulk")
    complete(scheduler, scheduler.schedule())
    grammar = caller_request(scheduler, "grammar", "bulk")
    scheduler.waiting.remove_request(grammar)
    grammar.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    grammar.structured_output_request = SimpleNamespace(grammar=object())
    scheduler.skipped_waiting.add_request(grammar)
    for i in range(5):
        caller_request(scheduler, f"a{i}", "bulk")
    if competing:
        caller_request(scheduler, "late", "occasional")
    step = scheduler.schedule()
    assert len(step.num_scheduled_tokens) <= 4
    assert scheduler._cold_prefill_ends.keys() == {
        "owner",
        "late" if competing else "grammar",
    }


def test_cached_competitor_allows_borrowing_and_bypasses_full_cap(monkeypatch):
    scheduler = caller_budget(monkeypatch, enable_prefix_caching=True)
    seed = caller_request(scheduler, "seed", "later", length=160)
    for _ in range(20):
        complete(scheduler, scheduler.schedule())
    scheduler.finish_requests(seed.request_id, RequestStatus.FINISHED_ABORTED)
    for i in range(9):
        req = caller_request(scheduler, f"a{i}", "a")
        req.skip_reading_prefix_cache = True
    cached = caller_request(scheduler, "cached", "later")
    step = scheduler.schedule()
    assert scheduler._cold_prefill_ends.keys() == {f"a{i}" for i in range(8)}
    assert cached.num_computed_tokens - step.num_scheduled_tokens["cached"] == 128


@pytest.mark.parametrize("state", [PauseState.PAUSED_NEW, PauseState.PAUSED_ALL])
def test_caller_share_respects_pause(monkeypatch, state):
    scheduler = caller_budget(monkeypatch, share=1)
    for name in ["a0", "a1", "b0"]:
        caller_request(scheduler, name, name[0])
    scheduler.set_pause_state(state)
    assert not scheduler.schedule().num_scheduled_tokens
    scheduler.set_pause_state(PauseState.UNPAUSED)
    assert list(scheduler.schedule().num_scheduled_tokens) == ["a0", "b0", "a1"]


def test_fair_share_requires_positive_limit(monkeypatch):
    with pytest.raises(ValueError, match="VLLM_COLD_PREFILL_CALLER_LIMIT"):
        caller_budget(monkeypatch, share=0)
