"""Tests for the lane detection module."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from config import CONFIG
from modules.lane_detection import LaneDetector, LaneResult
from simulation.sim_runner import SyntheticRoad


@pytest.fixture
def detector() -> LaneDetector:
    """Provide a lane detector configured from defaults."""
    return LaneDetector(CONFIG.lane)


def _draw_straight_lanes(offset: int = 0) -> np.ndarray:
    """Create a synthetic frame with two straight lane lines.

    Args:
        offset: Horizontal shift of both lanes (pixels) to simulate drift.
    """
    w, h = CONFIG.video.width, CONFIG.video.height
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (60, 60, 60)
    cx = w // 2 + offset
    cv2.line(img, (cx - 350, h), (cx - 40, int(h * 0.62)), (255, 255, 255), 10)
    cv2.line(img, (cx + 350, h), (cx + 40, int(h * 0.62)), (255, 255, 255), 10)
    return img


def test_detect_returns_laneresult(detector: LaneDetector) -> None:
    """detect() always returns a LaneResult with a frame centre."""
    frame = _draw_straight_lanes()
    result = detector.detect(frame)
    assert isinstance(result, LaneResult)
    assert result.frame_center_x == pytest.approx(CONFIG.video.width / 2)


def test_detects_centered_lanes(detector: LaneDetector) -> None:
    """Centred lanes are detected with a small offset."""
    frame = _draw_straight_lanes(offset=0)
    result = detector.detect(frame)
    assert result.detected
    assert abs(result.offset_frac) < CONFIG.lane.departure_caution_frac


def test_detects_drift(detector: LaneDetector) -> None:
    """Shifting the lanes produces a larger lateral offset."""
    centered = detector.detect(_draw_straight_lanes(offset=0))
    drifted = detector.detect(_draw_straight_lanes(offset=140))
    assert drifted.detected
    assert abs(drifted.offset_frac) > abs(centered.offset_frac)


def test_blank_frame_no_detection(detector: LaneDetector) -> None:
    """A featureless frame yields no lane detection and zero offset."""
    blank = np.zeros((CONFIG.video.height, CONFIG.video.width, 3), dtype=np.uint8)
    result = detector.detect(blank)
    assert not result.detected
    assert result.offset_frac == 0.0


def test_draw_does_not_mutate_input(detector: LaneDetector) -> None:
    """draw() must return a new image and leave the input unchanged."""
    frame = _draw_straight_lanes()
    original = frame.copy()
    result = detector.detect(frame)
    out = detector.draw(frame, result, departed=False)
    assert out.shape == frame.shape
    assert np.array_equal(frame, original)


def test_synthetic_road_frame_shape() -> None:
    """Synthetic road frames have the configured geometry."""
    road = SyntheticRoad(CONFIG)
    frame = road.frame(0)
    assert frame.shape == (CONFIG.video.height, CONFIG.video.width, 3)
