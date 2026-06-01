"""Heads-up display overlay rendering.

Draws a semi-transparent information panel in the top-left corner showing the
simulated ego speed, warning level (colour-coded), lane status, and TTC. When a
DANGER warning is flashing, a red border is drawn around the whole frame.
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

from config import HudConfig
from modules.warning_system import WarningLevel, WarningState

logger = logging.getLogger("adas.hud")

Color = Tuple[int, int, int]


class Hud:
    """Renders the ADAS heads-up display overlay."""

    def __init__(self, config: HudConfig) -> None:
        """Initialize the HUD renderer.

        Args:
            config: HUD configuration.
        """
        self._cfg = config
        self._level_colors = {
            WarningLevel.SAFE: config.color_safe,
            WarningLevel.CAUTION: config.color_caution,
            WarningLevel.DANGER: config.color_danger,
        }

    def render(
        self, frame: np.ndarray, state: WarningState, ego_speed_kmh: float
    ) -> np.ndarray:
        """Draw the HUD panel and any flash border on a copy of the frame.

        Args:
            frame: BGR image (not mutated).
            state: Current warning state.
            ego_speed_kmh: Simulated ego speed for display.

        Returns:
            New BGR image with the HUD overlay.
        """
        out = frame.copy()
        self._draw_panel(out, state, ego_speed_kmh)
        if state.flash_on:
            self._draw_flash_border(out)
        return out

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _draw_panel(
        self, frame: np.ndarray, state: WarningState, ego_speed_kmh: float
    ) -> None:
        """Blend a translucent panel and write the text lines onto it."""
        cfg = self._cfg
        x0, y0 = cfg.origin
        x1 = x0 + cfg.panel_width
        y1 = y0 + cfg.panel_height

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), cfg.panel_color, -1)
        cv2.addWeighted(overlay, cfg.panel_alpha, frame, 1 - cfg.panel_alpha, 0, frame)

        level_color: Color = self._level_colors[state.level]
        ttc_text = f"{state.ttc_s:.1f}s" if state.ttc_s is not None else "--"
        dist_text = (
            f"{state.nearest_distance_m:.1f}m"
            if state.nearest_distance_m is not None
            else "--"
        )
        lane_text = "DEPARTURE" if state.lane_departed else "centered"

        lines = [
            (f"SPEED   : {ego_speed_kmh:.0f} km/h", cfg.text_color),
            (f"WARNING : {state.level.label}", level_color),
            (f"LANE    : {lane_text}", cfg.text_color),
            (f"OBSTACLE: {dist_text}", cfg.text_color),
            (f"TTC     : {ttc_text}", cfg.text_color),
        ]

        tx = x0 + 12
        ty = y0 + cfg.line_height
        for text, color in lines:
            cv2.putText(
                frame,
                text,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                cfg.font_scale,
                color,
                cfg.font_thickness,
                cv2.LINE_AA,
            )
            ty += cfg.line_height

    def _draw_flash_border(self, frame: np.ndarray) -> None:
        """Draw a thick red border for DANGER flashing."""
        h, w = frame.shape[:2]
        cv2.rectangle(
            frame, (0, 0), (w - 1, h - 1), self._cfg.color_danger, thickness=14
        )
        cv2.putText(
            frame,
            "! DANGER !",
            (w // 2 - 110, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            self._cfg.color_danger,
            3,
            cv2.LINE_AA,
        )
