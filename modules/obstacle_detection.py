"""Obstacle detection: HOG pedestrian detector + MOG2 contour vehicle detector.

Two complementary detectors are combined:

* **Pedestrians** — OpenCV's built-in HOG + linear SVM
  (``cv2.HOGDescriptor`` with ``getDefaultPeopleDetector``).
* **Vehicles / moving blobs** — MOG2 background subtraction followed by
  morphological cleanup and contour extraction.

Each detection is annotated with an estimated distance using the pinhole
camera model ``d = (f * H) / h`` where *f* is focal length in pixels, *H* the
assumed real-world object height in metres, and *h* the bounding-box height in
pixels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import cv2
import numpy as np

from config import ObstacleConfig

logger = logging.getLogger("adas.obstacle")

Color = Tuple[int, int, int]


class ObstacleKind(Enum):
    """Category of a detected obstacle."""

    PEDESTRIAN = "pedestrian"
    VEHICLE = "vehicle"


@dataclass
class Obstacle:
    """A single detected obstacle.

    Attributes:
        kind: Pedestrian or vehicle.
        box: Bounding box as (x, y, w, h) in pixels.
        distance_m: Estimated distance in metres (pinhole model).
    """

    kind: ObstacleKind
    box: Tuple[int, int, int, int]
    distance_m: float

    @property
    def area(self) -> int:
        """Bounding-box area in pixels."""
        return self.box[2] * self.box[3]


class ObstacleDetector:
    """Detects pedestrians and vehicles and estimates their distance."""

    def __init__(self, config: ObstacleConfig) -> None:
        """Initialize detectors.

        Args:
            config: Obstacle detection configuration.
        """
        self._cfg = config

        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=config.mog2_history,
            varThreshold=config.mog2_var_threshold,
            detectShadows=config.mog2_detect_shadows,
        )
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, config.morph_kernel
        )
        logger.debug("ObstacleDetector initialised with %s", config)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> List[Obstacle]:
        """Detect all obstacles in a BGR frame.

        Args:
            frame: BGR image of shape (H, W, 3).

        Returns:
            List of :class:`Obstacle`, sorted nearest-first.
        """
        obstacles: List[Obstacle] = []
        obstacles.extend(self._detect_pedestrians(frame))
        obstacles.extend(self._detect_vehicles(frame))
        obstacles.sort(key=lambda o: o.distance_m)
        return obstacles

    def draw(self, frame: np.ndarray, obstacles: List[Obstacle]) -> np.ndarray:
        """Draw bounding boxes and distance labels on a copy of the frame.

        Args:
            frame: BGR image (not mutated).
            obstacles: Obstacles to draw.

        Returns:
            New BGR image with overlays.
        """
        out = frame.copy()
        for obs in obstacles:
            x, y, w, h = obs.box
            color: Color = (
                self._cfg.box_color_pedestrian
                if obs.kind is ObstacleKind.PEDESTRIAN
                else self._cfg.box_color_vehicle
            )
            cv2.rectangle(out, (x, y), (x + w, y + h), color, self._cfg.box_thickness)
            label = f"{obs.kind.value} {obs.distance_m:.1f}m"
            cv2.putText(
                out,
                label,
                (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return out

    # ------------------------------------------------------------------ #
    # Detectors
    # ------------------------------------------------------------------ #
    def _detect_pedestrians(self, frame: np.ndarray) -> List[Obstacle]:
        """Run the HOG + SVM pedestrian detector."""
        rects, _weights = self._hog.detectMultiScale(
            frame,
            winStride=self._cfg.hog_win_stride,
            padding=self._cfg.hog_padding,
            scale=self._cfg.hog_scale,
            hitThreshold=self._cfg.hog_hit_threshold,
        )
        results: List[Obstacle] = []
        for x, y, w, h in rects:
            distance = self._estimate_distance(h, self._cfg.pedestrian_real_height_m)
            results.append(
                Obstacle(
                    kind=ObstacleKind.PEDESTRIAN,
                    box=(int(x), int(y), int(w), int(h)),
                    distance_m=distance,
                )
            )
        return results

    def _detect_vehicles(self, frame: np.ndarray) -> List[Obstacle]:
        """Background subtraction + contour extraction for moving blobs."""
        mask = self._bg.apply(frame)
        # Drop MOG2 shadow pixels (value 127) keeping only hard foreground.
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        results: List[Obstacle] = []
        for contour in contours:
            if cv2.contourArea(contour) < self._cfg.min_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            distance = self._estimate_distance(h, self._cfg.vehicle_real_height_m)
            results.append(
                Obstacle(
                    kind=ObstacleKind.VEHICLE,
                    box=(x, y, w, h),
                    distance_m=distance,
                )
            )
        return results

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    def _estimate_distance(self, box_height_px: int, real_height_m: float) -> float:
        """Estimate distance via the pinhole camera model.

        ``d = (f * H) / h``

        Args:
            box_height_px: Bounding-box height in pixels.
            real_height_m: Assumed real-world object height in metres.

        Returns:
            Distance in metres, clamped to the configured range.
        """
        if box_height_px <= 0:
            return self._cfg.max_distance_m
        distance = (self._cfg.focal_length_px * real_height_m) / box_height_px
        return float(
            np.clip(distance, self._cfg.min_distance_m, self._cfg.max_distance_m)
        )
