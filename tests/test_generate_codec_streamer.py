"""Unit tests for `codec_streamer`: every codec row the talker produces is pushed to the
streamer as it is produced, and `end()` is called once generation finishes."""
import torch
from transformers.generation import StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

from tests.tiny_model import NUM_CODE_GROUPS, RecordingCriteria, fake_talker_result, tiny_model


_GREEDY = dict(max_new_tokens=6, do_sample=False, subtalker_dosample=False)


class _RecordingStreamer(BaseStreamer):
    def __init__(self) -> None:
        self.rows = []
        self.end_calls = 0

    def put(self, value):
        self.rows.append(value.clone())

    def end(self):
        self.end_calls += 1


def _run_generate(model, **extra):
    input_ids = [torch.arange(12).view(1, -1)]
    return model.generate(input_ids=input_ids, languages=["Auto"], **extra)


def test_streamer_receives_every_codec_row_in_order():
    model = tiny_model()
    streamer = _RecordingStreamer()

    codes_list, _ = _run_generate(model, codec_streamer=streamer, **_GREEDY)

    assert streamer.end_calls == 1
    assert all(row.shape == (1, NUM_CODE_GROUPS) for row in streamer.rows)
    assert torch.equal(torch.cat(streamer.rows, dim=0), codes_list[0])


def test_streamer_rows_match_returned_codes_when_stopped_early():
    model = tiny_model()
    streamer = _RecordingStreamer()
    criteria = StoppingCriteriaList([RecordingCriteria(stop=True)])

    codes_list, _ = _run_generate(model, codec_streamer=streamer, stopping_criteria=criteria, **_GREEDY)

    assert streamer.end_calls == 1
    assert codes_list[0].shape == (1, NUM_CODE_GROUPS)
    assert torch.equal(torch.cat(streamer.rows, dim=0), codes_list[0])


def test_streamer_is_forwarded_to_talker_generate_and_ended_once():
    model = tiny_model()
    calls = []

    def fake_talker_generate(**kwargs):
        calls.append(kwargs)
        return fake_talker_result(num_steps=3)

    model.talker.generate = fake_talker_generate
    streamer = _RecordingStreamer()

    _run_generate(model, codec_streamer=streamer)

    assert calls[0]["codec_streamer"] is streamer
    assert streamer.end_calls == 1


def test_talker_kwargs_have_no_codec_streamer_when_not_given():
    model = tiny_model()
    calls = []

    def fake_talker_generate(**kwargs):
        calls.append(kwargs)
        return fake_talker_result(num_steps=3)

    model.talker.generate = fake_talker_generate

    _run_generate(model)

    assert "codec_streamer" not in calls[0]
