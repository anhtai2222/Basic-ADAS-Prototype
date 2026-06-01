"""Synthetic road video generator and headless simulation runner.

The :class:`SyntheticRoad` class draws procedural road frames with OpenCV
primitives: a horizon, two swaying lane lines, dashed centre markings, and an
approaching obstacle block whose growing height drives the distance/TTC logic.

:func:`run_simulation` wires the synthetic frames through the full ADAS
pipeline (lane detection, obstacle detection, warning system, CAN, HUD) without
ever opening a display window, making it suitable for CI.
"""

from __future__ import annotations

import logging
import math
from typing import Iterator, Optional

import cv2
import numpy as np

from config import CONFIG, Config

logger = logging.getLogger("adas.sim")


class SyntheticRoad:
    """Generates procedural road frames for simulation/testing."""

    def __init__(self, config: Config = CONFIG) -> None:
        """Initialize the synthetic road generator.

        Args:
            config: Aggregate configuration.
        """
        self._cfg = config
        self._w = config.video.width
        self._h = config.video.height

    def frame(self, index: int) -> np.ndarray:
        """Render a single synthetic road frame.

        Args:
            index: Frame index (drives sway and obstacle growth).

        Returns:
            A BGR image of shape (H, W, 3).
        """
        sim = self._cfg.sim
        img = np.zeros((self._h, self._w, 3), dtype=np.uint8)

        horizon = int(self._h * 0.6)
        img[:horizon] = sim.sky_color
        img[horizon:] = sim.road_color

        sway = sim.sway_amplitude_px * math.sin(
            2 * math.pi * index / sim.sway_period_frames
        )
        center_top = self._w / 2 + sway * 0.3
        center_bottom = self._w / 2 + sway

        self._draw_lane_pair(img, horizon, center_top, center_bottom)
        self._draw_center_dashes(img, horizon, center_top, center_bottom, index)
        self._draw_obstacle(img, index, center_top)
        return img

    def frames(self) -> Iterator[np.ndarray]:
        """Yield all configured synthetic frames in order."""
        for i in range(self._cfg.sim.num_frames):
            yield self.frame(i)

    # ------------------------------------------------------------------ #
    # Drawing helpers
    # ------------------------------------------------------------------ #
    def _draw_lane_pair(
        self, img: np.ndarray, horizon: int, center_top: float, center_bottom: float
    ) -> None:
        """Draw the left and right solid lane boundaries."""
        sim = self._cfg.sim
        half_top = self._w * 0.06
        half_bottom = self._w * 0.40
        pairs = [
            (center_top - half_top, center_bottom - half_bottom),  # left
            (center_top + half_top, center_bottom + half_bottom),  # right
        ]
        for top_x, bottom_x in pairs:
            cv2.line(
                img,
                (int(bottom_x), self._h),
                (int(top_x), horizon),
                sim.lane_marking_color,
                10,
            )

    def _draw_center_dashes(
        self,
        img: np.ndarray,
        horizon: int,
        center_top: float,
        center_bottom: float,
        index: int,
    ) -> None:
        """Draw scrolling dashed centre-line markings."""
        sim = self._cfg.sim
        period = sim.dash_length + sim.dash_gap
        scroll = (index * 12) % period
        y = self._h
        while y > horizon:
            seg_top = max(horizon, y - sim.dash_length)
            t0 = (self._h - y) / (self._h - horizon)
            t1 = (self._h - seg_top) / (self._h - horizon)
            x0 = center_bottom + (center_top - center_bottom) * t0
            x1 = center_bottom + (center_top - center_bottom) * t1
            cv2.line(
                img,
                (int(x0), int(y - scroll)),
                (int(x1), int(seg_top - scroll)),
                sim.lane_marking_color,
                4,
            )
            y -= period

    def _draw_obstacle(self, img: np.ndarray, index: int, center_top: float) -> None:
        """Draw an approaching obstacle whose height grows over time."""
        sim = self._cfg.sim
        progress = index / max(1, self._cfg.sim.num_frames - 1)
        height = int(
            sim.obstacle_start_height
            + progress * (sim.obstacle_end_height - sim.obstacle_start_height)
        )
        width = int(height * 1.4)
        cx = int(center_top)
        top = int(self._h * 0.6)
        x0 = cx - width // 2
        y0 = top
        cv2.rectangle(
            img, (x0, y0), (x0 + width, y0 + height), sim.obstacle_color, -1
        )


def run_simulation(
    config: Config = CONFIG,
    show: bool = False,
    max_frames: Optional[int] = None,
) -> int:
    """Run the full ADAS pipeline over synthetic frames.

    Args:
        config: Aggregate configuration.
        show: If True, display frames in a window (non-CI use).
        max_frames: Optional cap on number of frames processed.

    Returns:
        Number of frames processed.
    """
    # Imported lazily so the synthetic generator can be used standalone.
    from modules.can_sim import CanSimulator
    from modules.hud import Hud
    from modules.lane_detection import LaneDetector
    from modules.obstacle_detection import ObstacleDetector, ObstacleKind
    from modules.warning_system import WarningSystem

    road = SyntheticRoad(config)
    lane_detector = LaneDetector(config.lane)
    obstacle_detector = ObstacleDetector(config.obstacle)
    warning_system = WarningSystem(config.warning)
    hud = Hud(config.hud)

    processed = 0
    with CanSimulator(config.can) as can_sim:
        for index, frame in enumerate(road.frames()):
            if max_frames is not None and index >= max_frames:
                break

            lane = lane_detector.detect(frame)
            obstacles = obstacle_detector.detect(frame)
            state = warning_system.evaluate(
                obstacles,
                lane.offset_frac,
                config.lane.departure_caution_frac,
                config.lane.departure_danger_frac,
            )

            nearest_kind = 0
            if obstacles:
                nearest = min(obstacles, key=lambda o: o.distance_m)
                nearest_kind = (
                    1 if nearest.kind is ObstacleKind.PEDESTRIAN else 2
                )
            can_sim.publish(state, lane.offset_frac, nearest_kind)

            if show:  # pragma: no cover - interactive only
                vis = lane_detector.draw(frame, lane, state.lane_departed)
                vis = obstacle_detector.draw(vis, obstacles)
                vis = hud.render(vis, state, config.video.ego_speed_kmh)
                cv2.imshow("ADAS Simulation", vis)
                if cv2.waitKey(int(1000 / config.video.fps)) & 0xFF == ord("q"):
                    break

            processed += 1

        if show:  # pragma: no cover
            cv2.destroyAllWindows()

    logger.info("Simulation finished: %d frames, %d CAN frames", processed, can_sim.frame_count)
    return processed


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    run_simulation(show=False)
