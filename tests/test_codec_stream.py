"""Unit tests for the streaming helpers in `qwen_tts.inference.codec_stream`."""
import threading
import time

import numpy as np
import pytest
import torch

from qwen_tts.inference.codec_stream import decode_in_chunks, iter_codec_rows


GROUPS = 4


def _row(value: int) -> torch.Tensor:
    return torch.full((GROUPS,), value, dtype=torch.long)


# --- iter_codec_rows ---


def test_rows_are_yielded_in_order_and_worker_is_joined():
    finished = threading.Event()

    def run_generate(streamer, stop_criteria):
        for i in range(5):
            streamer.put(_row(i).unsqueeze(0))
        streamer.end()
        finished.set()

    rows = list(iter_codec_rows(run_generate))

    assert [int(row[0]) for row in rows] == [0, 1, 2, 3, 4]
    assert finished.is_set()


def test_worker_exception_is_reraised_to_consumer_after_its_rows():
    def run_generate(streamer, stop_criteria):
        streamer.put(_row(0).unsqueeze(0))
        raise RuntimeError("talker failed")

    rows = iter_codec_rows(run_generate)

    assert int(next(rows)[0]) == 0
    with pytest.raises(RuntimeError, match="talker failed"):
        next(rows)


def test_closing_iterator_early_requests_stop_and_waits_for_worker():
    exited = threading.Event()
    stop_seen = threading.Event()

    def run_generate(streamer, stop_criteria):
        step = 0
        while True:
            input_ids = torch.zeros(1, step + 1, dtype=torch.long)
            if bool(stop_criteria(input_ids, None)[0]):
                stop_seen.set()
                break
            streamer.put(_row(step).unsqueeze(0))
            step += 1
            time.sleep(0.001)
        exited.set()

    rows = iter_codec_rows(run_generate)
    next(rows)
    next(rows)
    rows.close()

    assert stop_seen.is_set()
    assert exited.is_set()


# --- decode_in_chunks ---


class _FakeDecodeChunk:
    """Return one sample per new frame: the frame's first codebook value."""

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, codes: torch.Tensor, context_size: int) -> np.ndarray:
        self.calls.append((codes.clone(), context_size))
        return codes[context_size:, 0].to(torch.float32).numpy()


def test_rows_are_decoded_in_chunks_with_left_context_and_tail_flush():
    decode = _FakeDecodeChunk()

    audio = list(
        decode_in_chunks([_row(i) for i in range(7)], decode, chunk_frames=3, left_context_frames=2)
    )

    assert [chunk.tolist() for chunk in audio] == [[0, 1, 2], [3, 4, 5], [6]]
    assert [(codes.shape[0], context) for codes, context in decode.calls] == [(3, 0), (5, 2), (3, 2)]
    assert decode.calls[1][0][:2, 0].tolist() == [1, 2]
    assert decode.calls[2][0][:2, 0].tolist() == [4, 5]


def test_initial_context_seeds_the_first_chunk():
    decode = _FakeDecodeChunk()
    initial = torch.stack([_row(100), _row(101), _row(102)])

    audio = list(
        decode_in_chunks(
            [_row(0), _row(1)], decode, chunk_frames=2, left_context_frames=2, initial_context=initial
        )
    )

    assert audio[0].tolist() == [0, 1]
    codes, context = decode.calls[0]
    assert context == 2
    assert codes[:2, 0].tolist() == [101, 102]


def test_eos_row_ends_the_stream_without_being_decoded():
    decode = _FakeDecodeChunk()
    rows = [_row(0), _row(1), _row(9), _row(2)]

    audio = list(decode_in_chunks(rows, decode, chunk_frames=10, left_context_frames=0, eos_token_id=9))

    assert [chunk.tolist() for chunk in audio] == [[0, 1]]


def test_zero_left_context_never_passes_context():
    decode = _FakeDecodeChunk()

    list(decode_in_chunks([_row(i) for i in range(4)], decode, chunk_frames=2, left_context_frames=0))

    assert [(codes.shape[0], context) for codes, context in decode.calls] == [(2, 0), (2, 0)]


def test_invalid_chunk_frames_is_rejected():
    with pytest.raises(ValueError):
        list(decode_in_chunks([], _FakeDecodeChunk(), chunk_frames=0, left_context_frames=1))
