"""Unit tests for chunked / streaming decode of the 12Hz tokenizer decoder."""
import numpy as np
import pytest
import torch

from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
from tests.tiny_model import NUM_CODE_GROUPS, tiny_decoder


def _codes(num_frames: int) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, 32, (1, NUM_CODE_GROUPS, num_frames))


def test_decoder_output_length_is_exactly_frames_times_upsample():
    decoder = tiny_decoder()
    codes = _codes(11)

    with torch.inference_mode():
        wav = decoder(codes)

    assert wav.shape == (1, 1, 11 * decoder.total_upsample)


def test_decode_chunk_drops_only_the_context_audio():
    decoder = tiny_decoder()
    codes = _codes(7)

    with torch.inference_mode():
        full = decoder(codes)
        chunk = decoder.decode_chunk(codes, context_size=3)

    assert torch.equal(chunk, full[..., 3 * decoder.total_upsample :])


def test_chunked_decode_equals_sequential_decode_chunk_calls():
    decoder = tiny_decoder()
    codes = _codes(11)
    chunk_size, left_context = 4, 2

    with torch.inference_mode():
        chunked = decoder.chunked_decode(codes, chunk_size=chunk_size, left_context_size=left_context)
        pieces, start = [], 0
        while start < codes.shape[-1]:
            end = min(start + chunk_size, codes.shape[-1])
            context = left_context if start - left_context > 0 else start
            pieces.append(decoder.decode_chunk(codes[..., start - context : end], context))
            start = end
        sequential = torch.cat(pieces, dim=-1)

    assert torch.equal(chunked, sequential)
    assert sequential.shape[-1] == 11 * decoder.total_upsample


class _FakeTokenizerModel:
    def __init__(self, model_type: str) -> None:
        self.model_type = model_type
        self.calls = []

    def get_model_type(self):
        return self.model_type

    def decode_chunk(self, audio_codes, context_size):
        self.calls.append((audio_codes, context_size))
        new_frames = audio_codes.shape[1] - context_size
        return torch.arange(new_frames * 2, dtype=torch.float32).view(1, -1)


def test_inference_wrapper_decode_chunk_returns_float32_numpy_for_new_frames():
    tokenizer = Qwen3TTSTokenizer()
    tokenizer.model = _FakeTokenizerModel("qwen3_tts_tokenizer_12hz")
    tokenizer.device = torch.device("cpu")
    codes = np.zeros((5, NUM_CODE_GROUPS), dtype=np.int64)

    wav = tokenizer.decode_chunk(codes, context_size=2)

    passed_codes, passed_context = tokenizer.model.calls[0]
    assert passed_codes.shape == (1, 5, NUM_CODE_GROUPS) and passed_codes.dtype == torch.long
    assert passed_context == 2
    assert isinstance(wav, np.ndarray) and wav.dtype == np.float32 and wav.shape == (6,)


def test_inference_wrapper_decode_chunk_rejects_25hz_tokenizer():
    tokenizer = Qwen3TTSTokenizer()
    tokenizer.model = _FakeTokenizerModel("qwen3_tts_tokenizer_25hz")
    tokenizer.device = torch.device("cpu")

    with pytest.raises(ValueError, match="12Hz"):
        tokenizer.decode_chunk(torch.zeros(3, NUM_CODE_GROUPS, dtype=torch.long), context_size=0)
