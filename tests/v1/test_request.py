# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pickle
from array import array

import numpy as np
import pytest

from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import _with_packed_prompt
from vllm.v1.request import Request, RequestStatus, TokenBuffer
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


def test_token_buffer_sequence_operations():
    assert list(TokenBuffer()) == []
    tokens = TokenBuffer(iter([1, 70000, 129279]))
    assert tokens[-1] == 129279
    with pytest.raises(IndexError):
        _ = tokens[3]
    sliced = tokens[::-2]
    copied = tokens.copy()
    assert type(sliced) is list and sliced == [129279, 1]
    assert type(copied) is list and copied == [1, 70000, 129279]

    assert tokens == (1, 70000, 129279)
    assert tokens != (1, 129279, 70000)
    assert tokens + [7] == [1, 70000, 129279, 7]
    assert [7] + tokens == [7, 1, 70000, 129279]

    alias = tokens
    tokens += [3, 4]
    tokens[0] = 11
    assert tokens is alias
    assert tokens == [11, 70000, 129279, 3, 4]
    assert copied == [1, 70000, 129279]
    assert sliced == [129279, 1]
    assert isinstance(tokens, array) and tokens.itemsize == 4


@pytest.mark.parametrize(
    "dtype", [np.dtype("=i4"), np.dtype("=i4").newbyteorder("S"), np.dtype("=u4")]
)
def test_token_buffer_owns_strided_array_values(dtype: np.dtype):
    source = np.array([0, 99, 70000, 99, 2**31 - 1], dtype=dtype)
    tokens = TokenBuffer(source[::2])
    assert tokens == [0, 70000, 2**31 - 1]
    source[2] = 17
    tokens[0] = 42
    assert tokens[1] == 70000
    assert source[0] == 0


def test_token_buffer_rejects_out_of_range_ids():
    with pytest.raises(OverflowError):
        TokenBuffer([2**31])
    with pytest.raises(OverflowError):
        TokenBuffer(np.array([2**31], dtype=np.uint32))


@pytest.mark.parametrize(
    "invalid",
    [
        np.array([[1, 2], [3, 4]], dtype=np.int32),
        np.array([1, 2], dtype=np.int64),
        np.array([1, 2], dtype=np.float32),
    ],
    ids=["matrix", "int64", "float32"],
)
def test_token_buffer_rejects_invalid_arrays(invalid: np.ndarray):
    with pytest.raises(ValueError, match="1D array of 32-bit integer token IDs"):
        TokenBuffer(invalid)


@pytest.mark.parametrize("protocol", [4, 5])
def test_token_buffer_pickle_round_trip(protocol: int):
    original = TokenBuffer([1, 70000, 129279])
    restored = pickle.loads(pickle.dumps(original, protocol=protocol))
    assert type(restored) is TokenBuffer
    assert restored == original
    restored[0] = 42
    assert original[0] == 1


def test_token_buffer_pickle_owns_out_of_band_snapshot():
    tokens = TokenBuffer([1, 70000])
    frames: list[pickle.PickleBuffer] = []
    payload = pickle.dumps(tokens, protocol=5, buffer_callback=frames.append)
    (frame,) = frames
    with frame.raw() as view:
        receiver_frame = bytearray(view)
        tokens[0] = 42
        tokens.append(7)
        assert bytes(view) == receiver_frame
    restored = pickle.loads(payload, buffers=[receiver_frame])
    receiver_frame[:] = bytes(len(receiver_frame))
    assert type(restored) is TokenBuffer
    assert restored == [1, 70000]
