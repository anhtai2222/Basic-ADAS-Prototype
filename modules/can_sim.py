"""Virtual CAN bus simulator using python-can.

Broadcasts three ADAS signals as CAN frames on a virtual bus and mirrors every
transmitted frame to a text log:

================  ======  ==================================================
Signal            CAN ID  Payload encoding
================  ======  ==================================================
LANE_STATUS       0x100   byte0: status (0=ok,1=drift,2=departure)
                          byte1: |offset| as percent of width (0-100)
OBSTACLE_DIST     0x101   byte0-1: distance in cm, big-endian (uint16)
                          byte2: obstacle kind (0=none,1=ped,2=vehicle)
WARNING_LEVEL     0x102   byte0: level (0=safe,1=caution,2=danger)
                          byte1: TTC in tenths of a second (0-255)
================  ======  ==================================================

If python-can is not installed or a bus cannot be created the simulator falls
back to a log-only mode so the rest of the pipeline keeps running.
"""

from __future__ import annotations

import logging
from typing import Optional

from config import CanConfig
from modules.warning_system import WarningLevel, WarningState

logger = logging.getLogger("adas.can")


class CanSimulator:
    """Encodes ADAS state into CAN frames on a virtual python-can bus."""

    def __init__(self, config: CanConfig) -> None:
        """Initialize the virtual bus and open the log file.

        Args:
            config: CAN configuration.
        """
        self._cfg = config
        self._bus = None
        self._can = None
        self._log = open(config.log_path, "w", encoding="utf-8")
        self._log.write("timestamp_s,can_id,dlc,data_hex,decoded\n")
        self._frame_count = 0

        try:
            import can  # type: ignore

            self._can = can
            self._bus = can.interface.Bus(
                channel=config.channel,
                interface=config.interface,
                receive_own_messages=True,
            )
            logger.info(
                "Virtual CAN bus '%s' (%s) ready", config.channel, config.interface
            )
        except Exception as exc:  # pragma: no cover - env dependent
            logger.warning("python-can unavailable, log-only mode (%s)", exc)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def publish(
        self,
        state: WarningState,
        lane_offset_frac: float,
        nearest_kind: int,
    ) -> None:
        """Encode and transmit the three ADAS frames for one cycle.

        Args:
            state: Current warning state.
            lane_offset_frac: Signed lane offset as a fraction of width.
            nearest_kind: 0=none, 1=pedestrian, 2=vehicle.
        """
        self._send_lane_status(state, lane_offset_frac)
        self._send_obstacle_dist(state, nearest_kind)
        self._send_warning_level(state)

    def close(self) -> None:
        """Flush the log and shut down the bus."""
        try:
            self._log.flush()
            self._log.close()
        finally:
            if self._bus is not None:  # pragma: no cover
                self._bus.shutdown()
        logger.info("CAN simulator closed after %d frames", self._frame_count)

    def __enter__(self) -> "CanSimulator":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Frame encoders
    # ------------------------------------------------------------------ #
    def _send_lane_status(self, state: WarningState, offset_frac: float) -> None:
        if state.level is WarningLevel.DANGER and state.lane_departed:
            status = 2
        elif state.lane_departed:
            status = 1
        else:
            status = 0
        offset_pct = min(100, int(abs(offset_frac) * 100))
        data = bytes([status, offset_pct])
        self._transmit(
            self._cfg.id_lane_status,
            data,
            f"status={status} offset={offset_pct}%",
        )

    def _send_obstacle_dist(self, state: WarningState, kind: int) -> None:
        dist_cm = (
            int(min(state.nearest_distance_m, 655.0) * 100)
            if state.nearest_distance_m is not None
            else 0xFFFF
        )
        dist_cm = min(dist_cm, 0xFFFF)
        data = bytes([(dist_cm >> 8) & 0xFF, dist_cm & 0xFF, kind & 0xFF])
        self._transmit(
            self._cfg.id_obstacle_dist,
            data,
            f"dist={dist_cm}cm kind={kind}",
        )

    def _send_warning_level(self, state: WarningState) -> None:
        ttc_tenths = (
            min(255, int(state.ttc_s * 10)) if state.ttc_s is not None else 255
        )
        data = bytes([int(state.level), ttc_tenths])
        self._transmit(
            self._cfg.id_warning_level,
            data,
            f"level={state.level.label} ttc={ttc_tenths/10:.1f}s",
        )

    # ------------------------------------------------------------------ #
    # Transmission + logging
    # ------------------------------------------------------------------ #
    def _transmit(self, can_id: int, data: bytes, decoded: str) -> None:
        """Send one frame on the bus (if available) and log it."""
        timestamp = self._frame_count / 1000.0
        if self._can is not None and self._bus is not None:  # pragma: no cover
            msg = self._can.Message(
                arbitration_id=can_id, data=data, is_extended_id=False
            )
            try:
                self._bus.send(msg)
            except Exception as exc:
                logger.debug("CAN send failed: %s", exc)

        data_hex = data.hex(" ").upper()
        self._log.write(
            f"{timestamp:.3f},0x{can_id:03X},{len(data)},{data_hex},{decoded}\n"
        )
        self._frame_count += 1

    @property
    def frame_count(self) -> int:
        """Total frames transmitted/logged."""
        return self._frame_count
