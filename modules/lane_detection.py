"""Lane detection via Canny edge detection + Hough line transform.

Pipeline
--------
1. Convert to grayscale.
2. Gaussian blur to suppress noise.
3. Canny edge detection.
4. Mask a trapezoidal region of interest (ROI).
5. Probabilistic Hough transform to extract line segments.
6. Separate segments into left / right lanes by slope sign, fit an average
   line for each side, and estimate the lane centre.
7. Compare the lane centre against the frame centre to produce a lane-departure
   measurement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import LaneConfig

logger = logging.getLogger("adas.lane")

# Type aliases for clarity.
Line = Tuple[int, int, int, int]  # x1, y1, x2, y2
Color = Tuple[int, int, int]


@dataclass
class LaneResult:
    """Outcome of lane detection for a single frame.

    Attributes:
        left_line: Averaged left lane line (x1, y1, x2, y2) or None.
        right_line: Averaged right lane line (x1, y1, x2, y2) or None.
        lane_center_x: Estimated x of the lane centre at the frame bottom.
        frame_center_x: Geometric horizontal centre of the frame.
        offset_frac: Signed lateral offset as a fraction of frame width.
            Positive => vehicle is left of lane centre.
        detected: True if at least one lane line was found.
    """

    left_line: Optional[Line]
    right_line: Optional[Line]
    lane_center_x: Optional[float]
    frame_center_x: float
    offset_frac: float
    detected: bool


class LaneDetector:
    """Detects lane lines and measures lane departure."""

    def __init__(self, config: LaneConfig) -> None:
        """Initialize the detector.

        Args:
            config: Lane detection configuration.
        """
        self._cfg = config
        self._theta = np.deg2rad(config.hough_theta_deg)
        logger.debug("LaneDetector initialised with %s", config)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> LaneResult:
        """Run the full lane detection pipeline on a BGR frame.

        Args:
            frame: BGR image of shape (H, W, 3).

        Returns:
            A populated :class:`LaneResult`.
        """
        height, width = frame.shape[:2]
        edges = self._edges(frame)
        masked = self._region_of_interest(edges, width, height)
        segments = self._hough_lines(masked)
        left, right = self._average_lines(segments, width, height)

        frame_center_x = width / 2.0
        lane_center_x: Optional[float] = None
        offset_frac = 0.0

        if left is not None and right is not None:
            lane_center_x = (left[0] + right[0]) / 2.0
            offset_frac = (frame_center_x - lane_center_x) / width
        elif left is not None:
            offset_frac = (frame_center_x - left[0]) / width
        elif right is not None:
            offset_frac = (frame_center_x - right[0]) / width

        return LaneResult(
            left_line=left,
            right_line=right,
            lane_center_x=lane_center_x,
            frame_center_x=frame_center_x,
            offset_frac=offset_frac,
            detected=left is not None or right is not None,
        )

    def draw(self, frame: np.ndarray, result: LaneResult, departed: bool) -> np.ndarray:
        """Overlay detected lanes onto a copy of the frame.

        Args:
            frame: BGR image to draw on (not mutated).
            result: Lane detection result.
            departed: If True, lanes are drawn in the warning colour.

        Returns:
            New BGR image with lane overlay.
        """
        out = frame.copy()
        color: Color = (
            self._cfg.lane_color_warn if departed else self._cfg.lane_color_ok
        )
        for line in (result.left_line, result.right_line):
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(out, (x1, y1), (x2, y2), color, self._cfg.lane_thickness)

        if result.lane_center_x is not None:
            cx = int(result.lane_center_x)
            cv2.circle(out, (cx, out.shape[0] - 10), 6, color, -1)
        return out

    # ------------------------------------------------------------------ #
    # Pipeline stages
    # ------------------------------------------------------------------ #
    def _edges(self, frame: np.ndarray) -> np.ndarray:
        """Grayscale -> blur -> Canny edges."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(
            gray, self._cfg.gaussian_kernel, self._cfg.gaussian_sigma
        )
        return cv2.Canny(blurred, self._cfg.canny_low, self._cfg.canny_high)

    def _region_of_interest(
        self, edges: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        """Mask everything outside the configured trapezoidal ROI."""
        mask = np.zeros_like(edges)
        vertices = np.array(
            [
                [int(fx * width), int(fy * height)]
                for fx, fy in self._cfg.roi_vertices
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [vertices], 255)
        return cv2.bitwise_and(edges, mask)

    def _hough_lines(self, masked_edges: np.ndarray) -> List[Line]:
        """Probabilistic Hough transform returning a list of segments."""
        raw = cv2.HoughLinesP(
            masked_edges,
            rho=self._cfg.hough_rho,
            theta=self._theta,
            threshold=self._cfg.hough_threshold,
            minLineLength=self._cfg.hough_min_line_len,
            maxLineGap=self._cfg.hough_max_line_gap,
        )
        if raw is None:
            return []
        return [tuple(int(v) for v in seg[0]) for seg in raw]  # type: ignore[misc]

    def _average_lines(
        self, segments: List[Line], width: int, height: int
    ) -> Tuple[Optional[Line], Optional[Line]]:
        """Group segments by slope sign and fit one line per side.

        Returns:
            (left_line, right_line) each as (x_bottom, y_bottom, x_top, y_top)
            or None when no segments support that side.
        """
        left_fits: List[Tuple[float, float]] = []
        right_fits: List[Tuple[float, float]] = []

        for x1, y1, x2, y2 in segments:
            if x2 == x1:
                continue  # vertical, slope undefined
            slope = (y2 - y1) / (x2 - x1)
            if not (self._cfg.min_abs_slope <= abs(slope) <= self._cfg.max_abs_slope):
                continue
            intercept = y1 - slope * x1
            # In image coordinates y grows downward: left lane has negative slope.
            if slope < 0:
                left_fits.append((slope, intercept))
            else:
                right_fits.append((slope, intercept))

        y_bottom = height
        y_top = int(height * min(fy for _, fy in self._cfg.roi_vertices))

        left = self._fit_to_line(left_fits, y_bottom, y_top)
        right = self._fit_to_line(right_fits, y_bottom, y_top)
        return left, right

    @staticmethod
    def _fit_to_line(
        fits: List[Tuple[float, float]], y_bottom: int, y_top: int
    ) -> Optional[Line]:
        """Average (slope, intercept) pairs and project to endpoints."""
        if not fits:
            return None
        slope = float(np.mean([s for s, _ in fits]))
        intercept = float(np.mean([b for _, b in fits]))
        if abs(slope) < 1e-6:
            return None
        x_bottom = int((y_bottom - intercept) / slope)
        x_top = int((y_top - intercept) / slope)
        return (x_bottom, y_bottom, x_top, y_top)
