# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import pytest

import vllm.envs as envs
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


def test_preemption_releases_slot(monkeypatch):
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
def test_unrestricted_paths(monkeypatch, limit, kwargs):
    scheduler = budget(monkeypatch, limit, **kwargs)
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


def test_large_streaming_update_waits_for_slot(monkeypatch):
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
