"""Shared detection metadata in source RGB coordinates."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    kind: str
    box: tuple[float, float, float, float]
    score: float
