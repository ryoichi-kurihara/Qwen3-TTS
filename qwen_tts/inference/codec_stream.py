# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""Streaming helpers: pull codec rows out of a running `generate()` and decode them in chunks."""
import queue
import threading
from typing import Callable, Iterable, Iterator, List, Optional

import numpy as np
import torch
from transformers.generation import StoppingCriteria
from transformers.generation.streamers import BaseStreamer


class CodecQueueStreamer(BaseStreamer):
    """Collects the codec rows pushed by the talker so another thread can consume them."""

    _END = object()

    def __init__(self) -> None:
        self._queue: "queue.Queue" = queue.Queue()
        self._ended = False

    def put(self, value: torch.Tensor) -> None:
        # `value` is (batch_size, num_code_groups); streaming handles a single sequence.
        self._queue.put(value[0])

    def end(self) -> None:
        if not self._ended:
            self._ended = True
            self._queue.put(self._END)

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            item = self._queue.get()
            if item is self._END:
                return
            yield item


class StopRequestCriteria(StoppingCriteria):
    """Stops generation once `request_stop()` has been called."""

    def __init__(self) -> None:
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        return torch.full((input_ids.shape[0],), self._stop, dtype=torch.bool, device=input_ids.device)


def iter_codec_rows(
    run_generate: Callable[[BaseStreamer, StoppingCriteria], None],
) -> Iterator[torch.Tensor]:
    """
    Run `run_generate(streamer, stop_criteria)` in a worker thread and yield each codec row
    `(num_code_groups,)` as soon as the talker produces it.

    `run_generate` must pass `streamer` as `codec_streamer` and include `stop_criteria` in the
    `stopping_criteria` it hands to `generate()`. Closing the iterator early requests a stop
    through `stop_criteria` and waits for the worker, so the model is idle when this returns.
    An exception raised by the worker is re-raised to the consumer after the worker exits.
    """
    streamer = CodecQueueStreamer()
    stop_criteria = StopRequestCriteria()
    failure: List[BaseException] = []

    def worker() -> None:
        try:
            run_generate(streamer, stop_criteria)
        except BaseException as exc:  # forwarded to the consumer thread
            failure.append(exc)
        finally:
            streamer.end()

    thread = threading.Thread(target=worker, name="qwen3-tts-codec-stream", daemon=True)
    thread.start()
    try:
        for row in streamer:
            yield row
    finally:
        stop_criteria.request_stop()
        thread.join()
    if failure:
        raise failure[0]


def decode_in_chunks(
    rows: Iterable[torch.Tensor],
    decode_chunk: Callable[[torch.Tensor, int], np.ndarray],
    *,
    chunk_frames: int,
    left_context_frames: int,
    initial_context: Optional[torch.Tensor] = None,
    eos_token_id: Optional[int] = None,
) -> Iterator[np.ndarray]:
    """
    Group codec rows into chunks of `chunk_frames` and decode each chunk with up to
    `left_context_frames` preceding frames as left context.

    `decode_chunk(codes, context_size)` receives `(context_size + new_frames, num_code_groups)`
    and returns the waveform of the new frames. `initial_context` `(frames, num_code_groups)`
    seeds the context, e.g. with the reference codes of a voice clone prompt. A row whose first
    codebook equals `eos_token_id` ends the stream without being decoded, matching the
    truncation of the non-streaming `generate()`.
    """
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be >= 1")
    if left_context_frames < 0:
        raise ValueError("left_context_frames must be >= 0")

    context: Optional[torch.Tensor] = None
    if initial_context is not None and left_context_frames > 0 and initial_context.shape[0] > 0:
        context = initial_context[-left_context_frames:]
    pending: List[torch.Tensor] = []

    def flush() -> np.ndarray:
        nonlocal context
        new_codes = torch.stack(pending)
        pending.clear()
        if context is None:
            codes, context_size = new_codes, 0
        else:
            codes, context_size = torch.cat([context.to(new_codes.device), new_codes]), context.shape[0]
        audio = decode_chunk(codes, context_size)
        context = codes[-left_context_frames:] if left_context_frames > 0 else None
        return audio

    for row in rows:
        if eos_token_id is not None and int(row[0]) == eos_token_id:
            break
        pending.append(row)
        if len(pending) >= chunk_frames:
            yield flush()
    if pending:
        yield flush()
