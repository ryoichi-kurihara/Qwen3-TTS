"""Unit tests for forwarding `stopping_criteria` from
`Qwen3TTSForConditionalGeneration.generate()` to `self.talker.generate()`.

The forwarding tests replace the talker's `generate()` with a fake that records its
kwargs; the deferral tests run the real Hugging Face generation loop on the tiny model.

Run: python -m pytest -q tests
"""
import pytest
import torch
from transformers.generation import StoppingCriteriaList

from qwen_tts.core.models.modeling_qwen3_tts import _DeferredStoppingCriteria
from tests.tiny_model import (
    HIDDEN_SIZE,
    NUM_CODE_GROUPS,
    RecordingCriteria,
    fake_talker_result,
    tiny_model,
)


# Keyword arguments `generate()` passed to `talker.generate()` before
# `stopping_criteria` / `codec_streamer` forwarding was added. Must stay unchanged
# when the caller does not pass them.
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

_GREEDY = dict(max_new_tokens=6, do_sample=False, subtalker_dosample=False)


@pytest.fixture
def model_and_calls():
    model = tiny_model()
    calls = []

    def fake_talker_generate(**kwargs):
        calls.append(kwargs)
        return fake_talker_result(num_steps=3)

    model.talker.generate = fake_talker_generate
    return model, calls


def _run_generate(model, **extra):
    input_ids = [torch.arange(12).view(1, -1)]
    return model.generate(input_ids=input_ids, languages=["Auto"], **extra)


def test_stopping_criteria_is_forwarded_to_talker_generate_deferred(model_and_calls):
    model, calls = model_and_calls
    criteria = StoppingCriteriaList([RecordingCriteria(stop=False)])

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


def test_stop_requested_from_first_token_still_returns_one_codec_row():
    model = tiny_model()
    criteria = RecordingCriteria(stop=True)

    codes_list, hidden_list = _run_generate(
        model, stopping_criteria=StoppingCriteriaList([criteria]), **_GREEDY
    )

    # Caller's criteria is consulted only from the second sampled token onwards,
    # so the first token's codec row has been recorded when generation stops.
    assert criteria.lengths == [2]
    assert codes_list[0].shape == (1, NUM_CODE_GROUPS)
    assert hidden_list[0].shape == (1, HIDDEN_SIZE)


def test_non_stopping_criteria_does_not_change_greedy_output():
    codes_without, hidden_without = _run_generate(tiny_model(), **_GREEDY)
    criteria = RecordingCriteria(stop=False)
    codes_with, hidden_with = _run_generate(
        tiny_model(), stopping_criteria=StoppingCriteriaList([criteria]), **_GREEDY
    )

    assert criteria.lengths == [2, 3, 4, 5, 6]
    assert torch.equal(codes_with[0], codes_without[0])
    assert torch.equal(hidden_with[0], hidden_without[0])
