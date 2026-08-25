"""Manual check: streaming voice-clone generation on the real model.

Requires a GPU and the real model. Not collected by pytest.

Usage:
    python tests/manual/run_stream_check.py <ref_audio.wav> <ref_text.txt>

Env:
    QWEN3_TTS_MODEL  model id or path (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)
    OUTPUT_DIR       where WAVs are written (default: ./outputs/stream_check)
    CHUNK_FRAMES     frames per streamed chunk (default: 12, about 1 s)
    LEFT_CONTEXT     left context frames for chunked decode (default: 100)
    CANCEL_AFTER_S   seconds before the cancel event is set in case 3 (default: 1.5)

Cases, on one loaded model:
    1. stream: time to first chunk, total time, audio length (1_stream.wav)
    2. same codec rows decoded at once vs. chunk by chunk with left context:
       max abs diff and SNR of the chunked waveform against the full decode
       (2_full_decode.wav / 2_chunked_decode.wav)
    3. stream cancelled from another thread after CANCEL_AFTER_S seconds:
       must end soon after, and a following stream must succeed (3_after_cancel.wav)
    4. non-streaming generate_voice_clone still works afterwards (4_non_stream.wav)
"""
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers.generation import StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

import qwen_tts
from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.codec_stream import decode_in_chunks


TEXT = (
    "本日はお集まりいただきありがとうございます。これから、音声合成のストリーミング出力を確認するための、"
    "少し長めの日本語の文章を読み上げます。一文目では会議の目的を説明し、二文目では今後の予定を説明し、"
    "三文目では参加者への依頼事項を説明します。最後に、質疑応答の時間を十分に確保していますので、"
    "疑問点があれば遠慮なくお知らせください。それでは、順番に説明を始めます。"
)


class EventStoppingCriteria(StoppingCriteria):
    def __init__(self, event: threading.Event) -> None:
        self._event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self._event.is_set(), dtype=torch.bool, device=input_ids.device)


class RecordingStreamer(BaseStreamer):
    def __init__(self) -> None:
        self.rows = []

    def put(self, value):
        self.rows.append(value[0].clone())

    def end(self):
        pass


def run_stream(model, prompt, output_path: Path, chunk_frames: int, left_context: int, cancel_after_s=None):
    """Stream one generation; return (first_chunk_latency, elapsed, audio_seconds)."""
    event = threading.Event()
    timer = None
    kwargs = {}
    if cancel_after_s is not None:
        timer = threading.Timer(cancel_after_s, event.set)
        timer.start()
        kwargs["stopping_criteria"] = StoppingCriteriaList([EventStoppingCriteria(event)])

    chunks = []
    sample_rate = None
    first_latency = None
    started = time.perf_counter()
    for audio, sample_rate in model.generate_voice_clone_stream(
        text=TEXT, language="Japanese", voice_clone_prompt=prompt, chunk_frames=chunk_frames,
        left_context_frames=left_context, **kwargs
    ):
        if first_latency is None:
            first_latency = time.perf_counter() - started
        chunks.append(audio)
    elapsed = time.perf_counter() - started
    if timer is not None:
        timer.cancel()

    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    sf.write(str(output_path), audio, sample_rate)
    return first_latency, elapsed, len(audio) / sample_rate


def compare_decodes(model, prompt, output_dir: Path, chunk_frames: int, left_context: int):
    """Decode the same codec rows at once and chunk by chunk; return (max_abs_diff, snr_db)."""
    streamer = RecordingStreamer()
    input_ids, ref_ids, prompt_dict, languages = model._prepare_voice_clone_inputs(
        text=TEXT, language="Japanese", voice_clone_prompt=prompt
    )
    model.model.generate(
        input_ids=input_ids, ref_ids=ref_ids, voice_clone_prompt=prompt_dict, languages=languages,
        codec_streamer=streamer, **model._merge_generate_kwargs(),
    )
    codes = torch.stack(streamer.rows)
    ref_code = prompt_dict["ref_code"][0]
    tokenizer = model.model.speech_tokenizer
    upsample = int(tokenizer.get_decode_upsample_rate())

    wavs, _ = tokenizer.decode([{"audio_codes": torch.cat([ref_code.to(codes.device), codes])}])
    full = wavs[0][ref_code.shape[0] * upsample :]
    chunked = np.concatenate(list(decode_in_chunks(
        list(codes), tokenizer.decode_chunk, chunk_frames=chunk_frames, left_context_frames=left_context,
        initial_context=ref_code,
    )))
    assert full.shape == chunked.shape, (full.shape, chunked.shape)
    sample_rate = int(tokenizer.get_output_sample_rate())
    sf.write(str(output_dir / "2_full_decode.wav"), full, sample_rate)
    sf.write(str(output_dir / "2_chunked_decode.wav"), chunked, sample_rate)
    diff = full - chunked
    snr_db = 10 * np.log10(np.sum(full ** 2) / max(np.sum(diff ** 2), 1e-12))
    return float(np.max(np.abs(diff))), float(snr_db)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    ref_audio, ref_text_path = sys.argv[1], sys.argv[2]
    ref_text = Path(ref_text_path).read_text(encoding="utf-8").strip()
    model_id = os.environ.get("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    output_dir = Path(os.environ.get("OUTPUT_DIR", "outputs/stream_check"))
    chunk_frames = int(os.environ.get("CHUNK_FRAMES", "12"))
    left_context = int(os.environ.get("LEFT_CONTEXT", "100"))
    cancel_after_s = float(os.environ.get("CANCEL_AFTER_S", "1.5"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"qwen_tts: {qwen_tts.__file__}")
    print(f"model: {model_id}  chunk_frames: {chunk_frames}  left_context: {left_context}")
    model = Qwen3TTSModel.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
    )
    prompt = model.create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode=False)

    first, elapsed, audio_s = run_stream(model, prompt, output_dir / "1_stream.wav", chunk_frames, left_context)
    max_diff, snr_db = compare_decodes(model, prompt, output_dir, chunk_frames, left_context)
    _, cancel_elapsed, cancel_audio_s = run_stream(
        model, prompt, output_dir / "3_cancel.wav", chunk_frames, left_context, cancel_after_s
    )
    after_first, after_elapsed, after_audio_s = run_stream(
        model, prompt, output_dir / "3_after_cancel.wav", chunk_frames, left_context
    )
    started = time.perf_counter()
    wavs, sample_rate = model.generate_voice_clone(text=TEXT, language="Japanese", voice_clone_prompt=prompt)
    non_stream_elapsed = time.perf_counter() - started
    sf.write(str(output_dir / "4_non_stream.wav"), wavs[0], sample_rate)
    non_stream_audio_s = len(wavs[0]) / sample_rate

    print()
    print(f"{'case':22} {'first[s]':>9} {'elapsed[s]':>11} {'audio[s]':>9}")
    print(f"{'stream':22} {first:9.2f} {elapsed:11.2f} {audio_s:9.2f}")
    print(f"{'cancel (stream)':22} {'-':>9} {cancel_elapsed:11.2f} {cancel_audio_s:9.2f}")
    print(f"{'stream after cancel':22} {after_first:9.2f} {after_elapsed:11.2f} {after_audio_s:9.2f}")
    print(f"{'non-stream':22} {'-':>9} {non_stream_elapsed:11.2f} {non_stream_audio_s:9.2f}")
    print(f"chunked vs full decode: max|diff|={max_diff:.4f}  SNR={snr_db:.1f} dB")
    print(f"outputs: {output_dir}")

    checks = {
        "first chunk arrives within 2 s": first < 2.0,
        "stream keeps up with playback (elapsed < audio)": elapsed < audio_s,
        "chunked decode close to full decode (SNR > 20 dB)": snr_db > 20.0,
        "cancel ends close to CANCEL_AFTER_S": cancel_elapsed < cancel_after_s + 2.0,
        "cancelled stream is shorter than full stream": cancel_audio_s < audio_s * 0.6,
        "stream after cancel produced comparable audio": 0.7 < after_audio_s / audio_s < 1.3,
        "non-streaming generation still works": non_stream_audio_s > 0,
    }
    ok = True
    for name, passed in checks.items():
        print(f"[{'OK' if passed else 'NG'}] {name}")
        ok &= passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
