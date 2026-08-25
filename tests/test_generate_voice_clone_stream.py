"""Integration tests for `Qwen3TTSModel.generate_voice_clone_stream` with a fake core model:
kwargs threading (codec streamer, merged stopping criteria), reference-code context and
early close."""
import threading
import types

import numpy as np
import torch
from transformers.generation import StoppingCriteriaList

from qwen_tts.inference.codec_stream import StopRequestCriteria
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from tests.tiny_model import RecordingCriteria


GROUPS = 4
EOS = 9


class _FakeSpeechTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def get_model_type(self):
        return "qwen3_tts_tokenizer_12hz"

    def get_output_sample_rate(self):
        return 24000

    def decode_chunk(self, codes, context_size):
        self.calls.append((codes.clone(), context_size))
        return codes[context_size:, 0].to(torch.float32).numpy()


class _FakeCoreModel:
    """Pushes `num_rows` codec rows through the streamer, honouring the stopping criteria."""

    def __init__(self, num_rows: int) -> None:
        self.num_rows = num_rows
        self.speech_tokenizer = _FakeSpeechTokenizer()
        self.config = types.SimpleNamespace(talker_config=types.SimpleNamespace(codec_eos_token_id=EOS))
        self.tts_model_type = "base"
        self.device = torch.device("cpu")
        self.generate_config = {}
        self.calls = []
        self.finished = threading.Event()

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        streamer = kwargs["codec_streamer"]
        stop = kwargs["stopping_criteria"]
        for i in range(self.num_rows):
            if bool(stop(torch.zeros(1, i + 2, dtype=torch.long), None)[0]):
                break
            streamer.put(torch.full((1, GROUPS), i, dtype=torch.long))
        streamer.end()
        self.finished.set()
        return [torch.zeros(0, GROUPS)], [torch.zeros(0, GROUPS)]


def _wrapper(num_rows: int = 5, ref_code=None):
    core = _FakeCoreModel(num_rows)
    wrapper = Qwen3TTSModel(model=core, processor=None)
    wrapper._prepare_voice_clone_inputs = lambda **kwargs: (
        [torch.zeros(1, 12, dtype=torch.long)],
        None,
        {"ref_code": [ref_code]},
        ["Auto"],
    )
    return wrapper, core


def test_stream_yields_decoded_chunks_with_sample_rate():
    wrapper, core = _wrapper(num_rows=5)

    chunks = list(wrapper.generate_voice_clone_stream("text", chunk_frames=2, left_context_frames=0))

    assert [(audio.tolist(), sr) for audio, sr in chunks] == [([0, 1], 24000), ([2, 3], 24000), ([4], 24000)]
    assert core.finished.is_set()


def test_generate_receives_streamer_and_merged_stopping_criteria():
    wrapper, core = _wrapper(num_rows=3)
    caller = RecordingCriteria(stop=False)

    list(wrapper.generate_voice_clone_stream(
        "text", chunk_frames=10, left_context_frames=0, stopping_criteria=StoppingCriteriaList([caller])
    ))

    kwargs = core.calls[0]
    assert kwargs["codec_streamer"] is not None
    criteria = kwargs["stopping_criteria"]
    assert isinstance(criteria, StoppingCriteriaList)
    assert criteria[0] is caller
    assert isinstance(criteria[1], StopRequestCriteria)
    assert caller.lengths == [2, 3, 4]


def test_reference_codes_seed_the_left_context():
    ref_code = torch.full((6, GROUPS), 100, dtype=torch.long)
    wrapper, core = _wrapper(num_rows=2, ref_code=ref_code)

    chunks = list(wrapper.generate_voice_clone_stream("text", chunk_frames=2, left_context_frames=4))

    assert chunks[0][0].tolist() == [0, 1]
    codes, context = core.speech_tokenizer.calls[0]
    assert context == 4
    assert codes[:4, 0].tolist() == [100, 100, 100, 100]


def test_closing_the_stream_early_stops_generation_and_waits_for_it():
    wrapper, core = _wrapper(num_rows=1000)

    stream = wrapper.generate_voice_clone_stream("text", chunk_frames=1, left_context_frames=0)
    first_audio, _ = next(stream)
    stream.close()

    assert first_audio.tolist() == [0]
    assert core.finished.is_set()
