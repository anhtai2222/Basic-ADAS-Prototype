"""Warning system: TTC computation and a 3-level alert state machine.

Warning levels
--------------
* ``SAFE``    — no threat detected.
* ``CAUTION`` — obstacle inside the caution distance, moderate TTC, or mild
  lane drift.
* ``DANGER``  — obstacle inside the danger distance, TTC below the danger
  threshold, or large lane drift. Triggers visual flashing and (optionally) an
  audible beep.

Time-To-Collision is computed as ``TTC = distance / relative_speed``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

from config import WarningConfig
from modules.obstacle_detection import Obstacle

logger = logging.getLogger("adas.warning")


class WarningLevel(IntEnum):
    """Discrete warning severities (ordered)."""

    SAFE = 0
    CAUTION = 1
    DANGER = 2

    @property
    def label(self) -> str:
        """Human-readable label."""
        return self.name


@dataclass
class WarningState:
    """Aggregate warning state for a single frame.

    Attributes:
        level: Overall warning level for the frame.
        nearest_distance_m: Distance to the nearest obstacle, or None.
        ttc_s: Time-to-collision of the nearest obstacle, or None.
        lane_departed: Whether a lane-departure condition is active.
        reason: Short human-readable explanation of the level.
        flash_on: Whether a flashing overlay should currently be visible.
    """

    level: WarningLevel
    nearest_distance_m: Optional[float]
    ttc_s: Optional[float]
    lane_departed: bool
    reason: str
    flash_on: bool


class _Beeper:
    """Lazy, optional pygame-based beeper that degrades gracefully."""

    def __init__(self, config: WarningConfig) -> None:
        self._cfg = config
        self._available = False
        self._sound = None
        if not config.enable_audio:
            return
        try:  # pragma: no cover - environment dependent
            import numpy as np
            import pygame

            pygame.mixer.init(frequency=44100, size=-16, channels=1)
            sample_rate = 44100
            n_samples = int(sample_rate * config.beep_duration_ms / 1000)
            t = np.linspace(0, config.beep_duration_ms / 1000, n_samples, False)
            wave = 0.5 * np.sin(2 * np.pi * config.beep_frequency_hz * t)
            audio = (wave * 32767).astype(np.int16)
            self._sound = pygame.sndarray.make_sound(audio)
            self._available = True
            logger.debug("Beeper initialised (pygame)")
        except Exception as exc:  # pragma: no cover
            logger.warning("Audio disabled (%s)", exc)
            self._available = False

    def beep(self) -> None:
        """Play the beep if audio is available."""
        if self._available and self._sound is not None:  # pragma: no cover
            try:
                self._sound.play()
            except Exception as exc:
                logger.debug("Beep failed: %s", exc)


class WarningSystem:
    """Computes TTC and the overall warning level each frame."""

    def __init__(self, config: WarningConfig) -> None:
        """Initialize the warning system.

        Args:
            config: Warning configuration.
        """
        self._cfg = config
        self._beeper = _Beeper(config)
        self._last_beep_t = 0.0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def compute_ttc(self, distance_m: float, rel_speed_ms: Optional[float]) -> float:
        """Compute Time-To-Collision in seconds.

        Args:
            distance_m: Distance to the obstacle in metres.
            rel_speed_ms: Relative closing speed in m/s. If None or too small,
                the configured default / minimum is used.

        Returns:
            TTC in seconds, clamped to ``max_ttc_s``.
        """
        speed = rel_speed_ms if rel_speed_ms is not None else self._cfg.default_rel_speed_ms
        speed = max(speed, self._cfg.min_rel_speed_ms)
        ttc = distance_m / speed
        return min(ttc, self._cfg.max_ttc_s)

    def evaluate(
        self,
        obstacles: List[Obstacle],
        lane_offset_frac: float,
        lane_caution_frac: float,
        lane_danger_frac: float,
        rel_speed_ms: Optional[float] = None,
    ) -> WarningState:
        """Determine the overall warning level for a frame.

        Args:
            obstacles: Detected obstacles (need not be sorted).
            lane_offset_frac: Absolute lane offset as a fraction of width.
            lane_caution_frac: Lane offset that triggers CAUTION.
            lane_danger_frac: Lane offset that triggers DANGER.
            rel_speed_ms: Optional measured relative speed for TTC.

        Returns:
            A populated :class:`WarningState`.
        """
        nearest: Optional[Obstacle] = min(
            obstacles, key=lambda o: o.distance_m, default=None
        )
        nearest_dist = nearest.distance_m if nearest else None
        ttc = (
            self.compute_ttc(nearest.distance_m, rel_speed_ms) if nearest else None
        )

        level = WarningLevel.SAFE
        reasons: List[str] = []

        # Obstacle distance contribution.
        if nearest_dist is not None:
            if nearest_dist < self._cfg.danger_distance_m:
                level = max(level, WarningLevel.DANGER)
                reasons.append(f"obstacle {nearest_dist:.1f}m")
            elif nearest_dist < self._cfg.caution_distance_m:
                level = max(level, WarningLevel.CAUTION)
                reasons.append(f"obstacle {nearest_dist:.1f}m")

        # TTC contribution.
        if ttc is not None:
            if ttc < self._cfg.danger_ttc_s:
                level = max(level, WarningLevel.DANGER)
                reasons.append(f"TTC {ttc:.1f}s")
            elif ttc < self._cfg.caution_ttc_s:
                level = max(level, WarningLevel.CAUTION)
                reasons.append(f"TTC {ttc:.1f}s")

        # Lane departure contribution.
        abs_offset = abs(lane_offset_frac)
        lane_departed = abs_offset >= lane_caution_frac
        if abs_offset >= lane_danger_frac:
            level = max(level, WarningLevel.DANGER)
            reasons.append("lane departure")
        elif abs_offset >= lane_caution_frac:
            level = max(level, WarningLevel.CAUTION)
            reasons.append("lane drift")

        reason = ", ".join(reasons) if reasons else "clear"
        flash_on = self._flash_phase() if level is WarningLevel.DANGER else False

        if level is WarningLevel.DANGER:
            self._maybe_beep()

        return WarningState(
            level=level,
            nearest_distance_m=nearest_dist,
            ttc_s=ttc,
            lane_departed=lane_departed,
            reason=reason,
            flash_on=flash_on,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _flash_phase(self) -> bool:
        """Return True/False alternating at the configured flash rate."""
        phase = (time.time() * self._cfg.flash_hz) % 1.0
        return phase < 0.5

    def _maybe_beep(self) -> None:
        """Rate-limit beeps to roughly the flash rate."""
        now = time.time()
        if now - self._last_beep_t >= 1.0 / self._cfg.flash_hz:
            self._beeper.beep()
            self._last_beep_t = now
