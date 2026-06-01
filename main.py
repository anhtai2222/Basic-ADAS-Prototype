#!/usr/bin/env python3
"""ADAS prototype entry point.

Runs the Advanced Driver Assistance System pipeline against a webcam, a video
file, or a procedurally generated synthetic road (simulation mode).

Examples:
    python main.py --webcam
    python main.py --video assets/sample_video.mp4
    python main.py --simulate
    python main.py --simulate --headless --max-frames 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

import cv2
import numpy as np

from config import CONFIG, Config
from modules.can_sim import CanSimulator
from modules.hud import Hud
from modules.lane_detection import LaneDetector
from modules.obstacle_detection import ObstacleDetector, ObstacleKind
from modules.warning_system import WarningSystem

logger = logging.getLogger("adas.main")


def configure_logging(config: Config) -> None:
    """Set up the root logger from configuration.

    Args:
        config: Aggregate configuration.
    """
    logging.basicConfig(
        level=getattr(logging, config.logging.level, logging.INFO),
        format=config.logging.fmt,
        datefmt=config.logging.datefmt,
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Basic ADAS prototype (lane + obstacle + warning + CAN + HUD)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--webcam",
        action="store_true",
        help="Use the default system webcam as the video source.",
    )
    source.add_argument(
        "--video",
        type=str,
        metavar="PATH",
        help="Path to a video file to process.",
    )
    source.add_argument(
        "--simulate",
        action="store_true",
        help="Generate a synthetic road video instead of using a real source.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open any display window (for CI / servers).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames.",
    )
    return parser.parse_args(argv)


class AdasPipeline:
    """Wires together all ADAS modules and processes frames."""

    def __init__(self, config: Config = CONFIG) -> None:
        """Initialize all pipeline modules.

        Args:
            config: Aggregate configuration.
        """
        self._cfg = config
        self.lane_detector = LaneDetector(config.lane)
        self.obstacle_detector = ObstacleDetector(config.obstacle)
        self.warning_system = WarningSystem(config.warning)
        self.hud = Hud(config.hud)
        self.can_sim = CanSimulator(config.can)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the full pipeline.

        Args:
            frame: BGR input frame.

        Returns:
            Annotated BGR output frame (lanes + boxes + HUD).
        """
        lane = self.lane_detector.detect(frame)
        obstacles = self.obstacle_detector.detect(frame)
        state = self.warning_system.evaluate(
            obstacles,
            lane.offset_frac,
            self._cfg.lane.departure_caution_frac,
            self._cfg.lane.departure_danger_frac,
        )

        nearest_kind = 0
        if obstacles:
            nearest = min(obstacles, key=lambda o: o.distance_m)
            nearest_kind = 1 if nearest.kind is ObstacleKind.PEDESTRIAN else 2
        self.can_sim.publish(state, lane.offset_frac, nearest_kind)

        vis = self.lane_detector.draw(frame, lane, state.lane_departed)
        vis = self.obstacle_detector.draw(vis, obstacles)
        vis = self.hud.render(vis, state, self._cfg.video.ego_speed_kmh)
        return vis

    def close(self) -> None:
        """Release pipeline resources."""
        self.can_sim.close()


def _open_capture(args: argparse.Namespace, config: Config) -> cv2.VideoCapture:
    """Open a cv2.VideoCapture for webcam or file sources."""
    if args.webcam:
        cap = cv2.VideoCapture(args.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.video.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.video.height)
    else:
        cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video source.")
    return cap


def run_capture(args: argparse.Namespace, config: Config) -> int:
    """Run the pipeline against a webcam or video file.

    Args:
        args: Parsed CLI arguments.
        config: Aggregate configuration.

    Returns:
        Number of frames processed.
    """
    cap = _open_capture(args, config)
    pipeline = AdasPipeline(config)
    processed = 0
    try:
        while True:
            if args.max_frames is not None and processed >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (config.video.width, config.video.height))
            vis = pipeline.process(frame)
            processed += 1
            if not args.headless:
                cv2.imshow("ADAS Prototype", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        pipeline.close()
    logger.info("Processed %d frames", processed)
    return processed


def main(argv: Optional[list[str]] = None) -> int:
    """Program entry point.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    configure_logging(CONFIG)

    if args.simulate:
        from simulation.sim_runner import run_simulation

        run_simulation(
            CONFIG, show=not args.headless, max_frames=args.max_frames
        )
        return 0

    try:
        run_capture(args, CONFIG)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
