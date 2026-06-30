import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C
from frontend.asr import LiveTranscriber
from tacet.gate import should_respond
import voice_mngt.pitch as pitch_module
from voice_mngt.spectrogram import spectrogram

try:
    from tacet.infer import run_model_inference
except ModuleNotFoundError as exc:
    run_model_inference = None
    INFERENCE_IMPORT_ERROR = exc
else:
    INFERENCE_IMPORT_ERROR = None


def _threshold_seconds(value):
    return value / 1000 if value > 10 else value


def test_spectrogram_threshold():
    print("=" * 60)
    print("SPECTROGRAM THRESHOLD VERIFICATION")
    print("=" * 60)
    print(f"SPEC_THRESHOLD: {C.SPEC_THRESHOLD}")

    quiet_level = C.SPEC_THRESHOLD
    speech_level = C.SPEC_THRESHOLD + 0.01

    quiet = spectrogram(quiet_level)
    speech = spectrogram(speech_level)

    print(f"quiet level ({quiet_level:.3f}) -> {quiet}")
    print(f"speech level ({speech_level:.3f}) -> {speech}")

    assert quiet is False
    assert speech is True


def test_gate_timing_pipeline():
    print("\n" + "=" * 60)
    print("SPECTROGRAM + BERT GATE TIMING VERIFICATION")
    print("=" * 60)
    print(f"TAU: {C.TAU}")
    print(f"MIN_THRESHOLD: {C.MIN_THRESHOLD}")
    print(f"MAX_THRESHOLD: {C.MAX_THRESHOLD}")

    min_threshold = _threshold_seconds(C.MIN_THRESHOLD)
    max_threshold = _threshold_seconds(C.MAX_THRESHOLD)
    complete_prob = C.TAU + 0.1
    incomplete_prob = C.TAU - 0.1
    quiet_level = 0.0
    speech_level = C.SPEC_THRESHOLD + 0.01

    cases = [
        (
            "speech is active, keep listening",
            complete_prob,
            speech_level,
            max_threshold + 0.5,
            False,
        ),
        (
            "short silence, keep listening",
            complete_prob,
            quiet_level,
            min_threshold / 2,
            False,
        ),
        (
            "silence past minimum and complete, respond",
            complete_prob,
            quiet_level,
            min_threshold + 0.1,
            True,
        ),
        (
            "silence past minimum but incomplete, keep listening",
            incomplete_prob,
            quiet_level,
            min_threshold + 0.1,
            False,
        ),
        (
            "silence past maximum, respond",
            incomplete_prob,
            quiet_level,
            max_threshold + 0.1,
            True,
        ),
    ]

    for label, prob, audio_level, silence_duration, expected in cases:
        decision = should_respond(
            prob,
            audio_level=audio_level,
            silence_duration=silence_duration,
        )
        print(f"{label}: {decision}")
        assert decision is expected


def test_required_silence_pipeline():
    print("\n" + "=" * 60)
    print("DYNAMIC REQUIRED SILENCE VERIFICATION")
    print("=" * 60)

    cases = [
        ("400 ms boundary", C.TAU + 0.1, 0.400, 400, False),
        ("past 400 ms", C.TAU + 0.1, 0.401, 400, True),
        ("2000 ms boundary", C.TAU - 0.1, 2.000, 2000, False),
        ("past 2000 ms", C.TAU - 0.1, 2.001, 2000, True),
        ("active speech veto", C.TAU + 0.1, 3.000, 400, False),
    ]

    for label, prob, silence_duration, required_silence, expected in cases:
        audio_level = C.SPEC_THRESHOLD + 0.01 if label == "active speech veto" else 0.0
        decision = should_respond(
            prob,
            audio_level=audio_level,
            silence_duration=silence_duration,
            required_silence=required_silence,
        )
        print(f"{label}: {decision}")
        assert decision is expected


def test_pitch_shortens_silence_only_when_safe():
    print("\n" + "=" * 60)
    print("FRESH FALLING-PITCH GATE VERIFICATION")
    print("=" * 60)

    cases = [
        ("fresh falling pitch", C.TAU + 0.1, 0.201, 0.0, True, True, True),
        ("stale falling pitch", C.TAU + 0.1, 0.201, 0.0, True, False, False),
        ("incomplete text", C.TAU - 0.1, 0.201, 0.0, True, True, False),
        (
            "active speech veto",
            C.TAU + 0.1,
            0.500,
            C.SPEC_THRESHOLD + 0.01,
            True,
            True,
            False,
        ),
    ]
    for label, prob, silence, level, falling, fresh, expected in cases:
        decision = should_respond(
            prob,
            audio_level=level,
            silence_duration=silence,
            required_silence=800,
            pitch_falling=falling,
            pitch_fresh=fresh,
        )
        print(f"{label}: {decision}")
        assert decision is expected


def test_pitch_audio_window_and_worker_freshness():
    print("\n" + "=" * 60)
    print("ROLLING PITCH WINDOW VERIFICATION")
    print("=" * 60)

    transcriber = LiveTranscriber()
    transcriber._buf = [
        np.arange(0, 4, dtype=np.float32),
        np.arange(4, 8, dtype=np.float32),
        np.arange(8, 12, dtype=np.float32),
    ]
    transcriber._sample_count = 12
    transcriber._last_voice_sample = 9
    audio, token = transcriber.audio_window_before_last_speech(6 / 16000)
    assert token == 9
    assert audio.tolist() == [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    original_track_pitch = pitch_module.track_pitch
    completed = threading.Event()

    def fake_track_pitch(_audio):
        completed.set()
        return {
            "contour": [180.0, 150.0],
            "median": 165.0,
            "final": 150.0,
            "drop_ratio": 0.15,
            "falling": True,
        }

    pitch_module.track_pitch = fake_track_pitch
    tracker = pitch_module.RollingPitchTracker(
        lambda: (np.ones(3200, dtype=np.float32), 9),
        update_seconds=0.001,
    )
    try:
        tracker.start(warm_up=False)
        assert completed.wait(1.0)
        deadline = time.monotonic() + 1.0
        fresh = tracker.snapshot(current_token=9)
        while fresh is None and time.monotonic() < deadline:
            time.sleep(0.001)
            fresh = tracker.snapshot(current_token=9)
        stale = tracker.snapshot(current_token=10)
        assert fresh and fresh["fresh"] and fresh["falling"]
        assert stale and not stale["fresh"]
    finally:
        tracker.stop()
        pitch_module.track_pitch = original_track_pitch

    print("exact voiced window: True")
    print("newer-speech invalidation: True")


def test_inference_still_connects_to_gate():
    print("\n" + "=" * 60)
    print("MOCK BERT EMBEDDING -> INFERENCE -> GATE VERIFICATION")
    print("=" * 60)

    if run_model_inference is None:
        print(f"[SKIP] inference dependencies unavailable: {INFERENCE_IMPORT_ERROR}")
        return

    if not C.HEAD_PATH.exists():
        print(f"[SKIP] weights file not found at: {C.HEAD_PATH}")
        return

    mock_bert_vector = np.random.uniform(-1.0, 1.0, C.EMB_DIM).astype(np.float32)
    prob = run_model_inference(mock_bert_vector)
    decision = should_respond(
        prob,
        audio_level=0.0,
        silence_duration=_threshold_seconds(C.MIN_THRESHOLD) + 0.1,
    )

    print(f"P(complete): {prob:.3f}")
    print(f"gate decision: {decision}")
    assert isinstance(decision, bool)


def main():
    test_spectrogram_threshold()
    test_gate_timing_pipeline()
    test_required_silence_pipeline()
    test_pitch_shortens_silence_only_when_safe()
    test_pitch_audio_window_and_worker_freshness()
    test_inference_still_connects_to_gate()
    print("\nAll spectogram pipeline checks passed.")


if __name__ == "__main__":
    main()
