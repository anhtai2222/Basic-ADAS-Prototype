"""Centralized configuration for the ADAS prototype.

All tunable parameters, magic numbers, and thresholds live here so that the
rest of the codebase contains no hard-coded constants. Import the relevant
section (or the whole module) wherever values are needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    fmt: str = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
    datefmt: str = "%H:%M:%S"


# --------------------------------------------------------------------------- #
# Video / frame
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VideoConfig:
    """Video capture / synthetic frame geometry."""

    width: int = 960
    height: int = 540
    fps: int = 30
    # Simulated ego speed in km/h shown on the HUD and used for TTC.
    ego_speed_kmh: float = 60.0


# --------------------------------------------------------------------------- #
# Lane detection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LaneConfig:
    """Parameters for the Canny + Hough lane detection pipeline."""

    gaussian_kernel: Tuple[int, int] = (5, 5)
    gaussian_sigma: float = 0.0

    canny_low: int = 50
    canny_high: int = 150

    # Hough transform parameters.
    hough_rho: int = 2
    hough_theta_deg: float = 1.0          # degrees, converted to radians internally
    hough_threshold: int = 40
    hough_min_line_len: int = 40
    hough_max_line_gap: int = 100

    # Slope filtering: lines flatter than this are rejected (lane lines slope).
    min_abs_slope: float = 0.5
    max_abs_slope: float = 2.0

    # Region of interest as fractions of frame (x_frac, y_frac), bottom-up
    # trapezoid. Order: bottom-left, top-left, top-right, bottom-right.
    roi_vertices: Tuple[Tuple[float, float], ...] = (
        (0.05, 1.00),
        (0.43, 0.62),
        (0.57, 0.62),
        (0.95, 1.00),
    )

    # Lane departure: allowed deviation of lane centre from frame centre,
    # expressed as a fraction of frame width before a warning fires.
    departure_caution_frac: float = 0.08
    departure_danger_frac: float = 0.15

    # Drawing.
    lane_color_ok: Tuple[int, int, int] = (0, 255, 0)      # BGR green
    lane_color_warn: Tuple[int, int, int] = (0, 0, 255)    # BGR red
    lane_thickness: int = 8


# --------------------------------------------------------------------------- #
# Obstacle detection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ObstacleConfig:
    """Parameters for pedestrian (HOG) and vehicle (MOG2) detection."""

    # HOG pedestrian detector.
    hog_win_stride: Tuple[int, int] = (8, 8)
    hog_padding: Tuple[int, int] = (16, 16)
    hog_scale: float = 1.05
    hog_hit_threshold: float = 0.0

    # MOG2 background subtractor.
    mog2_history: int = 200
    mog2_var_threshold: float = 40.0
    mog2_detect_shadows: bool = True

    # Contour filtering for vehicle blobs.
    min_contour_area: int = 1500
    morph_kernel: Tuple[int, int] = (5, 5)

    # Pinhole distance estimation:  d = (focal_len_px * real_height_m) / box_h_px
    # Calibrated rough defaults for a 960x540 dashcam-style frame.
    focal_length_px: float = 700.0
    pedestrian_real_height_m: float = 1.7
    vehicle_real_height_m: float = 1.5

    # Clamp distance to a sane range (metres).
    min_distance_m: float = 1.0
    max_distance_m: float = 120.0

    # Drawing.
    box_color_pedestrian: Tuple[int, int, int] = (255, 128, 0)  # BGR orange-ish
    box_color_vehicle: Tuple[int, int, int] = (0, 200, 255)     # BGR amber
    box_thickness: int = 2


# --------------------------------------------------------------------------- #
# Warning system
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WarningConfig:
    """Thresholds for the 3-level warning state machine and TTC."""

    # Distance thresholds (metres).
    caution_distance_m: float = 5.0
    danger_distance_m: float = 3.0

    # TTC thresholds (seconds).
    danger_ttc_s: float = 2.0
    caution_ttc_s: float = 4.0

    # Assumed relative closing speed (m/s) when per-object speed is unknown.
    # Derived each frame from ego speed unless a measured value is supplied.
    default_rel_speed_ms: float = 16.6  # ~60 km/h

    # Avoid divide-by-zero / silly TTC values.
    min_rel_speed_ms: float = 0.5
    max_ttc_s: float = 99.0

    # DANGER alert flashing rate (Hz) and audio.
    flash_hz: float = 4.0
    enable_audio: bool = True
    beep_frequency_hz: int = 880
    beep_duration_ms: int = 150


# --------------------------------------------------------------------------- #
# CAN bus simulation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CanConfig:
    """Virtual CAN bus configuration and arbitration IDs."""

    channel: str = "adas_vbus"
    interface: str = "virtual"
    bitrate: int = 500_000

    id_lane_status: int = 0x100
    id_obstacle_dist: int = 0x101
    id_warning_level: int = 0x102

    log_path: str = "can_log.txt"


# --------------------------------------------------------------------------- #
# HUD overlay
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HudConfig:
    """Heads-up display overlay appearance."""

    origin: Tuple[int, int] = (15, 15)
    panel_width: int = 290
    panel_height: int = 150
    panel_alpha: float = 0.55
    panel_color: Tuple[int, int, int] = (0, 0, 0)

    text_color: Tuple[int, int, int] = (255, 255, 255)
    font_scale: float = 0.6
    font_thickness: int = 1
    line_height: int = 28

    color_safe: Tuple[int, int, int] = (0, 255, 0)
    color_caution: Tuple[int, int, int] = (0, 215, 255)
    color_danger: Tuple[int, int, int] = (0, 0, 255)


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SimConfig:
    """Synthetic road video generator parameters."""

    num_frames: int = 300
    road_color: Tuple[int, int, int] = (60, 60, 60)
    sky_color: Tuple[int, int, int] = (120, 90, 50)
    lane_marking_color: Tuple[int, int, int] = (255, 255, 255)
    dash_length: int = 40
    dash_gap: int = 40
    # Sinusoidal sway of the road centre to exercise the departure logic.
    sway_amplitude_px: float = 90.0
    sway_period_frames: float = 120.0
    # Synthetic obstacle block.
    obstacle_color: Tuple[int, int, int] = (40, 40, 200)
    obstacle_start_height: int = 30
    obstacle_end_height: int = 160


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    """Top-level aggregate configuration object."""

    logging: LoggingConfig = field(default_factory=LoggingConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    lane: LaneConfig = field(default_factory=LaneConfig)
    obstacle: ObstacleConfig = field(default_factory=ObstacleConfig)
    warning: WarningConfig = field(default_factory=WarningConfig)
    can: CanConfig = field(default_factory=CanConfig)
    hud: HudConfig = field(default_factory=HudConfig)
    sim: SimConfig = field(default_factory=SimConfig)


# Single shared default instance for convenient import.
CONFIG = Config()
