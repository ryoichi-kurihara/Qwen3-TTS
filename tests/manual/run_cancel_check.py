"""Manual check: cancel Qwen3-TTS generation from another thread via `stopping_criteria`.

Requires a GPU and the real model (downloaded on first use). Not collected by pytest.

Usage:
    python tests/manual/run_cancel_check.py <ref_audio.wav> <ref_text.txt>

Env:
    QWEN3_TTS_MODEL  model id or path (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)
    OUTPUT_DIR       where generated WAVs are written (default: ./outputs/cancel_check)
    CANCEL_AFTER_S   seconds before the cancel event is set (default: 1.5)

Checks, in order, on one loaded model:
    1. full generation (baseline time / audio length)
    2. generation cancelled from another thread after CANCEL_AFTER_S seconds:
       must finish sooner and return shorter audio than 1, without raising
    3. generation with the cancel event already set: shortest possible output,
       must decode without raising
    4. full generation again after the cancels: must succeed on the same model
       instance with time / audio length comparable to 1
"""
import os
import sys
import threading
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers.generation import StoppingCriteria, StoppingCriteriaList

import qwen_tts
from qwen_tts import Qwen3TTSModel


TEXT = (
    "本日はお集まりいただきありがとうございます。これから、音声合成の途中停止を確認するための、"
    "少し長めの日本語の文章を読み上げます。一文目では会議の目的を説明し、二文目では今後の予定を説明し、"
    "三文目では参加者への依頼事項を説明します。最後に、質疑応答の時間を十分に確保していますので、"
    "疑問点があれば遠慮なくお知らせください。それでは、順番に説明を始めます。"
)


class EventStoppingCriteria(StoppingCriteria):
    """Stop every sequence in the batch once `event` is set."""

    def __init__(self, event: threading.Event) -> None:
        self._event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full(
            (input_ids.shape[0],), self._event.is_set(), dtype=torch.bool, device=input_ids.device
        )


def generate(model, prompt, output_path: Path, cancel_after_s: float | None = None):
    """Run one voice-clone generation, optionally setting the cancel event after a delay."""
    event = threading.Event()
    timer = None
    if cancel_after_s is not None:
        if cancel_after_s <= 0:
            event.set()
        else:
            timer = threading.Timer(cancel_after_s, event.set)
            timer.start()
    kwargs = {}
    if cancel_after_s is not None:
        kwargs["stopping_criteria"] = StoppingCriteriaList([EventStoppingCriteria(event)])

    started = time.perf_counter()
    wavs, sample_rate = model.generate_voice_clone(
        text=TEXT, language="Japanese", voice_clone_prompt=prompt, **kwargs
    )
    elapsed = time.perf_counter() - started
    if timer is not None:
        timer.cancel()

    audio = wavs[0]
    sf.write(str(output_path), audio, sample_rate)
    return elapsed, len(audio) / sample_rate


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    ref_audio, ref_text_path = sys.argv[1], sys.argv[2]
    ref_text = Path(ref_text_path).read_text(encoding="utf-8").strip()
    model_id = os.environ.get("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    output_dir = Path(os.environ.get("OUTPUT_DIR", "outputs/cancel_check"))
    cancel_after_s = float(os.environ.get("CANCEL_AFTER_S", "1.5"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"qwen_tts: {qwen_tts.__file__}")
    print(f"model: {model_id}")
    model = Qwen3TTSModel.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
    )
    prompt = model.create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode=False)

    cases = [
        ("full", "1_full.wav", None),
        ("cancel", "2_cancel.wav", cancel_after_s),
        ("cancel_immediate", "3_cancel_immediate.wav", 0.0),
        ("full_again", "4_full_again.wav", None),
    ]
    results = {}
    errors = {}
    for name, filename, after_s in cases:
        try:
            results[name] = generate(model, prompt, output_dir / filename, after_s)
        except Exception as exc:  # report and keep going so later cases still run
            errors[name] = f"{type(exc).__name__}: {exc}"

    print()
    print(f"{'case':18} {'elapsed[s]':>10} {'audio[s]':>9}")
    for name, _, _ in cases:
        if name in results:
            elapsed, audio_s = results[name]
            print(f"{name:18} {elapsed:10.2f} {audio_s:9.2f}")
        else:
            print(f"{name:18} {'-':>10} {'-':>9}  {errors[name]}")
    print(f"outputs: {output_dir}")

    full_t, full_a = results["full"]
    cancel_t, cancel_a = results["cancel"]
    again_t, again_a = results["full_again"]
    checks = {
        "cancel finished sooner than full": cancel_t < full_t * 0.6,
        "cancel audio shorter than full": cancel_a < full_a * 0.6,
        "cancel elapsed close to CANCEL_AFTER_S": cancel_t < cancel_after_s + 2.0,
        "cancel_immediate did not raise": "cancel_immediate" in results,
        "full_again audio comparable to full": 0.7 < again_a / full_a < 1.3,
    }
    ok = True
    for name, passed in checks.items():
        print(f"[{'OK' if passed else 'NG'}] {name}")
        ok &= passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
