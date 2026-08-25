"""Unit tests for forwarding `stopping_criteria` from
`Qwen3TTSForConditionalGeneration.generate()` to `self.talker.generate()`.

A tiny randomly initialised model is built on CPU (no weights download). The
forwarding tests replace the talker's `generate()` with a fake that records its
kwargs; the deferral tests run the real Hugging Face generation loop.

Run: python -m pytest -q tests
"""
import types

import pytest
import torch
from transformers.generation import StoppingCriteria, StoppingCriteriaList

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSForConditionalGeneration,
    _DeferredStoppingCriteria,
)


NUM_CODE_GROUPS = 4
HIDDEN_SIZE = 16
CODEC_EOS_TOKEN_ID = 2150

# Keyword arguments `generate()` passed to `talker.generate()` before
# `stopping_criteria` forwarding was added. Must stay unchanged when the
# caller does not pass `stopping_criteria`.
BASELINE_TALKER_KWARGS = {
    "inputs_embeds",
    "attention_mask",
    "trailing_text_hidden",
    "tts_pad_embed",
    "max_new_tokens",
    "min_new_tokens",
    "do_sample",
    "top_k",
    "top_p",
    "temperature",
    "subtalker_dosample",
    "subtalker_top_k",
    "subtalker_top_p",
    "subtalker_temperature",
    "eos_token_id",
    "repetition_penalty",
    "suppress_tokens",
    "output_hidden_states",
    "return_dict_in_generate",
}


class _RecordingCriteria(StoppingCriteria):
    """Return `stop` for every sequence and record the `input_ids` length of each call."""

    def __init__(self, stop: bool) -> None:
        self.stop = stop
        self.lengths: list[int] = []

    def __call__(self, input_ids, scores, **kwargs):
        self.lengths.append(input_ids.shape[1])
        return torch.full((input_ids.shape[0],), self.stop, dtype=torch.bool, device=input_ids.device)


def _tiny_config() -> Qwen3TTSConfig:
    return Qwen3TTSConfig(
        tts_model_type="custom_voice",
        tts_pad_token_id=101,
        tts_bos_token_id=102,
        tts_eos_token_id=103,
        talker_config=dict(
            vocab_size=3072,
            text_vocab_size=200,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            text_hidden_size=HIDDEN_SIZE,
            num_code_groups=NUM_CODE_GROUPS,
            # mrope sections must sum to half the head dim (16 / 2 heads / 2 = 4).
            rope_scaling={"interleaved": True, "mrope_section": [2, 1, 1], "rope_type": "default", "type": "default"},
            codec_eos_token_id=CODEC_EOS_TOKEN_ID,
            codec_pad_id=2148,
            codec_bos_id=2149,
            codec_think_id=2154,
            codec_nothink_id=2155,
            codec_think_bos_id=2156,
            codec_think_eos_id=2157,
            codec_language_id={"japanese": 2058},
            spk_id={},
            spk_is_dialect={},
            code_predictor_config=dict(
                vocab_size=2048,
                hidden_size=HIDDEN_SIZE,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                num_code_groups=NUM_CODE_GROUPS,
            ),
        ),
    )


def _fake_talker_result(num_steps: int):
    """Mimic `talker.generate()` output: per-step `(layer_hidden_states, codec_ids)`.

    Step 0 is the prefill step (no codec ids); the last step emits the codec EOS.
    """
    def hidden():
        return (torch.zeros(1, 1, HIDDEN_SIZE),)

    def codes(first):
        return torch.tensor([[first] + [7] * (NUM_CODE_GROUPS - 1)])

    steps = [(hidden(), None)]
    steps += [(hidden(), codes(1)) for _ in range(num_steps - 1)]
    steps.append((hidden(), codes(CODEC_EOS_TOKEN_ID)))
    return types.SimpleNamespace(hidden_states=tuple(steps))


def _tiny_model() -> Qwen3TTSForConditionalGeneration:
    torch.manual_seed(0)
    return Qwen3TTSForConditionalGeneration(_tiny_config()).eval()


@pytest.fixture
def model_and_calls():
    model = _tiny_model()
    calls = []

    def fake_talker_generate(**kwargs):
        calls.append(kwargs)
        return _fake_talker_result(num_steps=3)

    model.talker.generate = fake_talker_generate
    return model, calls


def _run_generate(model, **extra):
    input_ids = [torch.arange(12).view(1, -1)]
    return model.generate(input_ids=input_ids, languages=["Auto"], **extra)


def test_stopping_criteria_is_forwarded_to_talker_generate_deferred(model_and_calls):
    model, calls = model_and_calls
    criteria = StoppingCriteriaList([_RecordingCriteria(stop=False)])

    _run_generate(model, stopping_criteria=criteria)

    assert len(calls) == 1
    assert set(calls[0]) == BASELINE_TALKER_KWARGS | {"stopping_criteria"}
    forwarded = calls[0]["stopping_criteria"]
    assert isinstance(forwarded, StoppingCriteriaList) and len(forwarded) == 1
    assert isinstance(forwarded[0], _DeferredStoppingCriteria)
    assert forwarded[0].criteria is criteria
    assert forwarded[0].min_new_tokens == calls[0]["min_new_tokens"] == 2


def test_talker_kwargs_and_return_shape_unchanged_without_stopping_criteria(model_and_calls):
    model, calls = model_and_calls

    codes_list, hidden_list = _run_generate(model)

    assert len(calls) == 1
    assert "stopping_criteria" not in calls[0]
    assert set(calls[0]) == BASELINE_TALKER_KWARGS
    # Return shape: one entry per input, codes truncated at the codec EOS step.
    assert isinstance(codes_list, list) and isinstance(hidden_list, list)
    assert len(codes_list) == len(hidden_list) == 1
    assert codes_list[0].shape == (2, NUM_CODE_GROUPS)
    assert hidden_list[0].shape == (2, HIDDEN_SIZE)


# --- real Hugging Face generation loop on the tiny model ---

_GREEDY = dict(max_new_tokens=6, do_sample=False, subtalker_dosample=False)


def test_stop_requested_from_first_token_still_returns_one_codec_row():
    model = _tiny_model()
    criteria = _RecordingCriteria(stop=True)

    codes_list, hidden_list = _run_generate(
        model, stopping_criteria=StoppingCriteriaList([criteria]), **_GREEDY
    )

    # Caller's criteria is consulted only from the second sampled token onwards,
    # so the first token's codec row has been recorded when generation stops.
    assert criteria.lengths == [2]
    assert codes_list[0].shape == (1, NUM_CODE_GROUPS)
    assert hidden_list[0].shape == (1, HIDDEN_SIZE)


def test_non_stopping_criteria_does_not_change_greedy_output():
    codes_without, hidden_without = _run_generate(_tiny_model(), **_GREEDY)
    criteria = _RecordingCriteria(stop=False)
    codes_with, hidden_with = _run_generate(
        _tiny_model(), stopping_criteria=StoppingCriteriaList([criteria]), **_GREEDY
    )

    assert criteria.lengths == [2, 3, 4, 5, 6]
    assert torch.equal(codes_with[0], codes_without[0])
    assert torch.equal(hidden_with[0], hidden_without[0])
