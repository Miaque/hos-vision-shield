"""Shared frame contract and explicit quality/lightweight/legacy model selection."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    kind: str
    box: tuple[float, float, float, float]
    score: float


class Detector(Protocol):
    def detect(self, rgb: np.ndarray) -> list[tuple[float, float, float, float]]: ...


def load_detector(config: dict) -> Detector:
    if config["detection_backend"] == "quality":
        from vision_shield.detection.quality import Detector as QualityDetector
        return QualityDetector(config)
    if config["detection_backend"] == "cpu":
        from vision_shield.detection.cpu import Detector as Implementation
    elif config["detection_backend"] == "egoblur":
        from vision_shield.detection.egoblur import Detector as Implementation
    else:
        raise ValueError("Unsupported detection backend")
    return Implementation(config["face_model_path"], config["plate_model_path"],
                          config["detection_max_side"], config["detection_box_padding"])
