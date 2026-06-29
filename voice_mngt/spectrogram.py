from __future__ import annotations
from typing import Protocol, runtime_checkable
from config import SPEC_THRESHOLD


@runtime_checkable
class HasLevel(Protocol):
    def level(self) -> float:
        ...


def is_above_threshold(value: float, threshold: float = SPEC_THRESHOLD) -> bool:
    """Return True when the provided spectrogram/RMS value exceeds threshold."""

    return float(value) > threshold


def asr_level_above_threshold(transcriber: HasLevel, threshold: float = SPEC_THRESHOLD) -> bool:
    """Read the current ASR input level and compare it with SPEC_THRESHOLD."""

    if not isinstance(transcriber, HasLevel):
        raise TypeError("transcriber must expose a level() method like LiveTranscriber")
    return is_above_threshold(transcriber.level(), threshold)


def spectrogram(input_value: float | HasLevel, threshold: float = SPEC_THRESHOLD) -> bool:
    """Return True if the provided value or ASR transcriber level is above threshold."""

    if isinstance(input_value, HasLevel):
        return asr_level_above_threshold(input_value, threshold)
    return is_above_threshold(input_value, threshold)
