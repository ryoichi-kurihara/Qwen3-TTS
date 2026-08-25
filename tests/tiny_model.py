"""Tiny randomly initialised models for CPU tests (no weights download)."""
import types

import torch
from transformers.generation import StoppingCriteria

from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Decoder


NUM_CODE_GROUPS = 4
HIDDEN_SIZE = 16
CODEC_EOS_TOKEN_ID = 2150


class RecordingCriteria(StoppingCriteria):
    """Return `stop` for every sequence and record the `input_ids` length of each call."""

    def __init__(self, stop: bool) -> None:
        self.stop = stop
        self.lengths: list[int] = []

    def __call__(self, input_ids, scores, **kwargs):
        self.lengths.append(input_ids.shape[1])
        return torch.full((input_ids.shape[0],), self.stop, dtype=torch.bool, device=input_ids.device)


def tiny_config() -> Qwen3TTSConfig:
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


def tiny_model() -> Qwen3TTSForConditionalGeneration:
    torch.manual_seed(0)
    return Qwen3TTSForConditionalGeneration(tiny_config()).eval()


def fake_talker_result(num_steps: int):
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


def tiny_decoder_config() -> Qwen3TTSTokenizerV2DecoderConfig:
    return Qwen3TTSTokenizerV2DecoderConfig(
        codebook_size=32,
        hidden_size=16,
        latent_dim=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        codebook_dim=16,
        sliding_window=4,
        intermediate_size=32,
        num_hidden_layers=1,
        num_quantizers=NUM_CODE_GROUPS,
        upsample_rates=(2, 2),
        upsampling_ratios=(2,),
        decoder_dim=16,
    )


def tiny_decoder() -> Qwen3TTSTokenizerV2Decoder:
    torch.manual_seed(0)
    return Qwen3TTSTokenizerV2Decoder._from_config(tiny_decoder_config()).eval()
