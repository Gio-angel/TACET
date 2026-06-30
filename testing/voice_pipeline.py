import sys
import time
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C
from frontend.asr import LiveTranscriber
from tacet.encoder import BertEncoder
from tacet.gate import should_respond
from tacet.infer import TacetEndpointer
from testing.simulation.results import calculate_metrics, print_metrics
from voice_mngt.pitch import PITCH_WINDOW_SECONDS, RollingPitchTracker
from voice_mngt.spectrogram import spectrogram

WHISPER_DIR = ROOT / "models" / "whisper"
BERT_DIR = ROOT / "models" / "bert"


def _threshold_seconds(value):
    return value / 1000 if value > 10 else value


class VoicePipelineRunner:
    def __init__(self):
        self.lock = Lock()
        self.raw_text = ""
        self.committed_text = ""
        self.raw_changed_at = 0.0
        self.last_voice_at = 0.0
        self.turn_started_at = 0.0
        self.last_encoded = ""
        self.last_prob = 0.0
        self.required_silence = 2000
        self.last_pitch = None
        self.pitch_calls = 0
        self.pitch_latency_ms = None
        self.pitch_error = ""
        self.pitch_sample = None
        self.bert_calls = 0
        self.decision_made = False
        self.decision_reason = ""
        self.rows = []

        self.encoder = BertEncoder(model_dir=BERT_DIR)
        self.endpointer = None
        self.transcriber = LiveTranscriber(
            on_partial=self.on_partial,
            on_display=self.on_display,
            on_final=self.on_final,
            on_status=self.on_status,
            model_size=C.WHISPER_MODEL_SIZE,
            model_dir=WHISPER_DIR,
        )
        self.pitch_tracker = RollingPitchTracker(
            lambda: self.transcriber.audio_window_before_last_speech(PITCH_WINDOW_SECONDS),
            token_source=self.transcriber.last_voice_sample,
        )

    def on_status(self, text):
        print(f"[status] {text}")

    def on_display(self, text):
        now = time.monotonic()
        with self.lock:
            if text != self.raw_text:
                self.raw_text = text
                self.raw_changed_at = now
        print(f"\rASR: {text[:100]}", end="", flush=True)

    def on_partial(self, text):
        with self.lock:
            self.committed_text = text

    def on_final(self, text):
        with self.lock:
            self.raw_text = text or self.raw_text

    def load_models(self):
        print("[status] loading BERT...")
        self.encoder.ensure_loaded(status=lambda text: print(f"[status] {text}"))
        print("[status] loading TACET head...")
        self.endpointer = TacetEndpointer()

    def encode_if_needed(self, text):
        if not text or text == self.last_encoded:
            return self.last_prob
        if self.endpointer is None:
            self.endpointer = TacetEndpointer()
        started = time.monotonic()
        emb = self.encoder.encode(text)
        self.last_prob = self.endpointer.probability(emb)
        self.last_encoded = text
        self.bert_calls += 1
        elapsed = (time.monotonic() - started) * 1000
        print(f"\n[bert] P(complete)={self.last_prob:.3f} ({elapsed:.1f} ms)")
        return self.last_prob

    def refresh_pitch(self):
        state = self.pitch_tracker.snapshot(self.transcriber.last_voice_sample())
        if state is None:
            return None
        self.pitch_calls = self.pitch_tracker.call_count
        if state["voice_sample"] == self.pitch_sample:
            return state

        self.pitch_sample = state["voice_sample"]
        self.pitch_latency_ms = state.get("latency_ms")
        self.pitch_error = state.get("error", "")
        self.last_pitch = state if "contour" in state else None
        if self.pitch_error:
            print(f"\n[pitch] error: {self.pitch_error}")
        elif self.last_pitch is None:
            latency = self.pitch_latency_ms or 0.0
            print(f"\n[pitch] no voiced F0 detected ({latency:.1f} ms)")
        else:
            contour = self.last_pitch["contour"]
            print(
                "\n[pitch] "
                f"median={self.last_pitch['median']:.2f} Hz "
                f"final={self.last_pitch['final']:.2f} Hz "
                f"drop={self.last_pitch['drop_ratio']:.1%} "
                f"falling={self.last_pitch['falling']} "
                f"frames={len(contour)} ({self.pitch_latency_ms:.1f} ms)"
            )

        return state

    def run(self):
        self.load_models()
        self.turn_started_at = time.monotonic()
        self.raw_changed_at = self.turn_started_at
        self.last_voice_at = self.turn_started_at

        print("[status] starting microphone. Speak now; Ctrl+C to stop.")
        print("[status] warming up rolling pitch tracker...")
        try:
            self.pitch_tracker.start()
            self.transcriber.start()
            while not self.decision_made:
                time.sleep(0.05)
                now = time.monotonic()
                audio_level = self.transcriber.level()

                if spectrogram(audio_level):
                    self.last_voice_at = now
                callback_voice_at = self.transcriber.last_voice_time()
                if callback_voice_at:
                    self.last_voice_at = max(self.last_voice_at, callback_voice_at)

                pitch_state = self.refresh_pitch()

                with self.lock:
                    raw = self.raw_text

                if raw and raw != self.last_encoded:
                    prob = self.encode_if_needed(raw)
                    self.required_silence = 400 if prob > C.TAU else 2000
                    continue

                silence_duration = now - self.last_voice_at
                if not raw:
                    continue

                bert_gate = should_respond(
                    self.last_prob,
                    audio_level=audio_level,
                    silence_duration=silence_duration,
                    required_silence=self.required_silence,
                    pitch_falling=bool(pitch_state and pitch_state.get("falling")),
                    pitch_fresh=bool(pitch_state and pitch_state.get("fresh")),
                )
                spectrogram_gate = should_respond(
                    0.0,
                    audio_level=audio_level,
                    silence_duration=silence_duration,
                    required_silence=2000,
                )

                if bert_gate:
                    self.decision_reason = "asr_bert_tacet_gate"
                    self.answer(now, raw, silence_duration)
                elif spectrogram_gate:
                    self.decision_reason = "asr_spectrogram_gate"
                    self.answer(now, raw, silence_duration)
        except KeyboardInterrupt:
            print("\n[status] stopped by user")
            self.decision_reason = "manual_stop"
            self.log_row(time.monotonic(), self.raw_text, 0.0)
        finally:
            self.pitch_tracker.stop()
            final_text = self.transcriber.stop()
            if final_text:
                with self.lock:
                    self.raw_text = final_text
            self.print_report()

    def answer(self, decision_time, transcript, silence_duration):
        self.decision_made = True
        print("\nANSWER: Gate returned True. Responding now.")
        print("\nPOUTI ARITO")
        self.log_row(decision_time, transcript, silence_duration)

    def log_row(self, decision_time, transcript, silence_duration):
        self.rows.append(
            {
                "decision": "auto_response" if self.decision_made else "manual_stop",
                "decision_reason": self.decision_reason,
                "transcript": transcript,
                "bert_prob": f"{self.last_prob:.6f}",
                "bert_calls": str(self.bert_calls),
                "pitch_median_hz": self._pitch_value("median"),
                "pitch_final_hz": self._pitch_value("final"),
                "pitch_drop_ratio": self._pitch_value("drop_ratio"),
                "pitch_falling": str(bool(self.last_pitch and self.last_pitch["falling"])),
                "pitch_frames": str(len(self.last_pitch["contour"])) if self.last_pitch else "0",
                "pitch_calls": str(self.pitch_calls),
                "pitch_latency_ms": (
                    f"{self.pitch_latency_ms:.3f}" if self.pitch_latency_ms is not None else ""
                ),
                "pitch_error": self.pitch_error,
                "turn_start_time": str(self.turn_started_at),
                "decision_time": str(decision_time),
                "decision_offset": str(decision_time - self.turn_started_at),
                "silence_duration": str(silence_duration),
                "gt_user_done_offset": "",
            }
        )

    def _pitch_value(self, key):
        if self.last_pitch is None:
            return ""
        return f"{self.last_pitch[key]:.6f}"

    def print_report(self):
        print("\n")
        metrics = calculate_metrics(self.rows)
        print_metrics(metrics)
        print("\nPipeline summary")
        print(f"Final transcript: {self.raw_text}")
        print(f"BERT calls:       {self.bert_calls}")
        print(f"Last probability: {self.last_prob:.3f}")
        print(f"YIN calls:        {self.pitch_calls}")
        if self.last_pitch:
            print(f"Median pitch:     {self.last_pitch['median']:.2f} Hz")
            print(f"Final pitch:      {self.last_pitch['final']:.2f} Hz")
            print(f"Pitch frames:     {len(self.last_pitch['contour'])}")
        else:
            print("Pitch result:     unavailable")
        if self.pitch_latency_ms is not None:
            print(f"YIN latency:      {self.pitch_latency_ms:.1f} ms")
        if self.pitch_error:
            print(f"Pitch error:      {self.pitch_error}")
        print(f"Decision reason:  {self.decision_reason or 'none'}")


def main():
    runner = VoicePipelineRunner()
    runner.run()


if __name__ == "__main__":
    main()
