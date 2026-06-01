"""Tests for the obstacle detection and warning system modules."""

from __future__ import annotations

import numpy as np
import pytest

from config import CONFIG
from modules.obstacle_detection import Obstacle, ObstacleDetector, ObstacleKind
from modules.warning_system import WarningLevel, WarningSystem


@pytest.fixture
def detector() -> ObstacleDetector:
    """Provide an obstacle detector configured from defaults."""
    return ObstacleDetector(CONFIG.obstacle)


@pytest.fixture
def warning() -> WarningSystem:
    """Provide a warning system with audio disabled for tests."""
    cfg = CONFIG.warning
    # Use a copy with audio off so tests never touch the sound device.
    import dataclasses

    return WarningSystem(dataclasses.replace(cfg, enable_audio=False))


# --------------------------------------------------------------------------- #
# Distance estimation
# --------------------------------------------------------------------------- #
def test_distance_inverse_to_height(detector: ObstacleDetector) -> None:
    """A taller bounding box must yield a smaller estimated distance."""
    near = detector._estimate_distance(200, CONFIG.obstacle.vehicle_real_height_m)
    far = detector._estimate_distance(50, CONFIG.obstacle.vehicle_real_height_m)
    assert near < far


def test_distance_matches_pinhole_formula(detector: ObstacleDetector) -> None:
    """Distance equals f*H/h within the clamp range."""
    h = 100
    expected = (
        CONFIG.obstacle.focal_length_px * CONFIG.obstacle.vehicle_real_height_m / h
    )
    got = detector._estimate_distance(h, CONFIG.obstacle.vehicle_real_height_m)
    assert got == pytest.approx(expected, rel=1e-6)


def test_distance_clamped(detector: ObstacleDetector) -> None:
    """Extreme box heights are clamped to the configured range."""
    tiny = detector._estimate_distance(1, CONFIG.obstacle.vehicle_real_height_m)
    huge = detector._estimate_distance(100000, CONFIG.obstacle.vehicle_real_height_m)
    assert tiny == pytest.approx(CONFIG.obstacle.max_distance_m)
    assert huge == pytest.approx(CONFIG.obstacle.min_distance_m)


def test_static_scene_no_moving_vehicles(detector: ObstacleDetector) -> None:
    """After MOG2 warms up on a static scene, no vehicle blobs are reported.

    On the very first frame MOG2 has no model and flags everything as
    foreground, so we feed several identical frames to let the background model
    converge, then assert the static scene produces no vehicle detections.
    """
    blank = np.zeros((CONFIG.video.height, CONFIG.video.width, 3), dtype=np.uint8)
    for _ in range(CONFIG.obstacle.mog2_history):
        detector.detect(blank)
    vehicles = [o for o in detector.detect(blank) if o.kind is ObstacleKind.VEHICLE]
    assert vehicles == []


# --------------------------------------------------------------------------- #
# TTC
# --------------------------------------------------------------------------- #
def test_ttc_basic(warning: WarningSystem) -> None:
    """TTC = distance / relative speed."""
    assert warning.compute_ttc(20.0, 10.0) == pytest.approx(2.0)


def test_ttc_handles_zero_speed(warning: WarningSystem) -> None:
    """A near-zero relative speed is floored, not divided by zero."""
    ttc = warning.compute_ttc(10.0, 0.0)
    assert ttc <= CONFIG.warning.max_ttc_s
    assert ttc > 0


def test_ttc_capped(warning: WarningSystem) -> None:
    """Very large distances produce a capped TTC."""
    ttc = warning.compute_ttc(10_000.0, CONFIG.warning.min_rel_speed_ms)
    assert ttc == pytest.approx(CONFIG.warning.max_ttc_s)


# --------------------------------------------------------------------------- #
# Warning levels
# --------------------------------------------------------------------------- #
def _obstacle(distance: float) -> Obstacle:
    return Obstacle(kind=ObstacleKind.VEHICLE, box=(0, 0, 50, 50), distance_m=distance)


def test_safe_when_clear(warning: WarningSystem) -> None:
    """No obstacles and centred lane => SAFE."""
    state = warning.evaluate([], 0.0, 0.08, 0.15, rel_speed_ms=10.0)
    assert state.level is WarningLevel.SAFE
    assert not state.lane_departed


def test_caution_on_medium_obstacle(warning: WarningSystem) -> None:
    """An obstacle just inside the caution distance => at least CAUTION."""
    state = warning.evaluate([_obstacle(4.0)], 0.0, 0.08, 0.15, rel_speed_ms=1.0)
    assert state.level >= WarningLevel.CAUTION


def test_danger_on_close_obstacle(warning: WarningSystem) -> None:
    """A very close obstacle => DANGER."""
    state = warning.evaluate([_obstacle(2.0)], 0.0, 0.08, 0.15, rel_speed_ms=10.0)
    assert state.level is WarningLevel.DANGER


def test_danger_on_low_ttc(warning: WarningSystem) -> None:
    """Low TTC at a moderate distance still escalates to DANGER."""
    state = warning.evaluate([_obstacle(6.0)], 0.0, 0.08, 0.15, rel_speed_ms=10.0)
    # 6m / 10 m/s = 0.6s < danger_ttc_s
    assert state.level is WarningLevel.DANGER


def test_lane_departure_triggers_warning(warning: WarningSystem) -> None:
    """Large lane offset alone triggers DANGER and lane_departed."""
    state = warning.evaluate([], 0.2, 0.08, 0.15, rel_speed_ms=10.0)
    assert state.lane_departed
    assert state.level is WarningLevel.DANGER
