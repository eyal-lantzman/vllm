# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle

from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import _with_packed_prompt
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


def test_request_status_fmt_str():
    """Test that the string representation of RequestStatus is correct."""
    assert f"{RequestStatus.WAITING}" == "WAITING"
    assert (
        f"{RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR}"
        == "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR"
    )
    assert f"{RequestStatus.WAITING_FOR_REMOTE_KVS}" == "WAITING_FOR_REMOTE_KVS"
    assert f"{RequestStatus.WAITING_FOR_STREAMING_REQ}" == "WAITING_FOR_STREAMING_REQ"
    assert f"{RequestStatus.RUNNING}" == "RUNNING"
    assert f"{RequestStatus.PREEMPTED}" == "PREEMPTED"
    assert f"{RequestStatus.FINISHED_STOPPED}" == "FINISHED_STOPPED"
    assert f"{RequestStatus.FINISHED_LENGTH_CAPPED}" == "FINISHED_LENGTH_CAPPED"
    assert f"{RequestStatus.FINISHED_ABORTED}" == "FINISHED_ABORTED"
    assert f"{RequestStatus.FINISHED_IGNORED}" == "FINISHED_IGNORED"


def test_request_copies_session_id_from_engine_core_request():
    engine_request = EngineCoreRequest(
        request_id="request-1",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        session_id="session-1",
    )

    request = Request.from_engine_core_request(engine_request, block_hasher=None)

    assert request.session_id == "session-1"


def test_packed_prompt_owns_tokens_across_engine_admission():
    tokens = [1001, 2002] * 4096
    frontend = EngineCoreRequest(
        request_id="packed",
        prompt_token_ids=tokens,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    outbound = _with_packed_prompt(frontend)
    tokens[0] = 3003
    received = MsgpackDecoder(EngineCoreRequest).decode(
        MsgpackEncoder().encode(outbound)
    )
    request = Request.from_engine_core_request(received, block_hasher=None)
    assert frontend.prompt_token_ids is tokens
    assert request.prompt_token_ids[0] == 1001

    frames: list[pickle.PickleBuffer] = []
    snapshot = pickle.dumps(
        request._all_token_ids, protocol=5, buffer_callback=frames.append
    )
    request.append_output_token_ids(4004)
    assert list(pickle.loads(snapshot, buffers=frames)) == [1001, 2002] * 4096
    assert request.all_token_ids[-1] == 4004
    assert len(request.prompt_token_ids) == 8192
