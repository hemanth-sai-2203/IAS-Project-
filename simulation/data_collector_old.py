"""
data_collection.py
------------------
Collects 2000 synchronized frames (500 × 4 weather states) from CARLA 0.9.15.
Saves Camera, LiDAR, RADAR, and ground-truth labels to disk.

Windows note: run from the ra_asf project root, not from inside simulation/
    cd C:\\Users\\heman\\Music\\ra_asf
    python simulation\\data_collector.py
"""

import os
import sys
import json
import queue
import logging
import argparse
import time
import carla

import cv2
import numpy as np
from tqdm import tqdm
import math # Make sure this is imported at the top of your file!

def spectator_follow(world, vehicle):
    """Snaps the CARLA spectator camera 10m behind and 6m above the vehicle."""
    transform = vehicle.get_transform()
    yaw = math.radians(transform.rotation.yaw)
    
    world.get_spectator().set_transform(carla.Transform(
        carla.Location(
            x=transform.location.x - 10 * math.cos(yaw),
            y=transform.location.y - 10 * math.sin(yaw),
            z=transform.location.z + 6,
        ),
        carla.Rotation(pitch=-20, yaw=transform.rotation.yaw),
    ))

# ── Path setup: allow running as script or as module ─────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import (
    DATA_ROOT, WEATHER_STATES, FRAMES_PER_WEATHER, DEGRADATION,
)
from simulation.carla_setup import CarlaSetup
from simulation.weather_engine import WeatherEngine

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── DEGRADATION ───────────────────────────────────────────────────────────────

def apply_camera_degradation(img, state):
    """Gaussian blur to simulate rain / fog on lens. img: uint8 (H,W,3)."""
    sigma = DEGRADATION[state]["blur_sigma"]
    if sigma <= 0.0:
        return img
    ksize = int(6 * sigma + 1) | 1     # must be odd
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def apply_lidar_degradation(points, state):
    """Random dropout + intensity scaling. points: float32 (N,4)."""
    if points.shape[0] == 0:
        return points
    dropout = DEGRADATION[state]["lidar_dropout"]
    scale   = DEGRADATION[state]["lidar_intensity_scale"]
    result  = points.copy()
    result[:, 3] *= scale
    if dropout > 0.0:
        keep = np.random.random(len(result)) > dropout
        result = result[keep]
    return result


# ── SAVING ────────────────────────────────────────────────────────────────────

def create_directories():
    for weather in WEATHER_STATES:
        for sub in ["images", "lidar", "radar", "labels"]:
            os.makedirs(os.path.join(DATA_ROOT, weather, sub), exist_ok=True)
    logger.info("Output directories ready: %s", DATA_ROOT)


def save_frame(frame_id, weather, img, lidar_pts, radar_pts, labels):
    """
    Save one synchronized frame.
    Layout: data/collected/<weather>/{images,lidar,radar,labels}/frame_XXXXXX.*
    """
    base  = os.path.join(DATA_ROOT, weather)
    fname = "frame_{:06d}".format(frame_id)

    # Image (.jpg, quality 95)
    img_path = os.path.join(base, "images", fname + ".jpg")
    cv2.imwrite(img_path,
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])

    # LiDAR float32 .npy
    np.save(os.path.join(base, "lidar", fname + ".npy"), lidar_pts)

    # RADAR float32 .npy
    np.save(os.path.join(base, "radar", fname + ".npy"), radar_pts)

    # Labels .json
    with open(os.path.join(base, "labels", fname + ".json"), "w") as f:
        json.dump({
            "frame_id"  : frame_id,
            "weather"   : weather,
            "num_points": int(lidar_pts.shape[0]),
            "num_radar" : int(radar_pts.shape[0]),
            "objects"   : labels,
        }, f, indent=2)


# ── WARM-UP ───────────────────────────────────────────────────────────────────

def warm_up(setup, n_ticks=30):
    """Discard first n_ticks frames — CARLA needs ticks to stabilise."""
    logger.info("Warming up (%d ticks) ...", n_ticks)
    for _ in range(n_ticks):
        try:
            setup.tick()
        except queue.Empty:
            pass


# ── MAIN ──────────────────────────────────────────────────────────────────────

def collect():
    create_directories()
    setup = CarlaSetup()

    try:
        setup.connect()
        setup.spawn_vehicle()
        setup.attach_sensors()

        engine      = WeatherEngine(setup.world)
        total       = 0

        for weather in WEATHER_STATES:
            logger.info("=" * 56)
            logger.info("Weather: %-12s  target: %d frames", weather, FRAMES_PER_WEATHER)
            logger.info("=" * 56)

            engine.set_weather(weather)
            warm_up(setup, n_ticks=30)

            collected  = 0
            errors     = 0
            MAX_ERRORS = 20

            with tqdm(total=FRAMES_PER_WEATHER, desc=weather, unit="fr") as pbar:
                while collected < FRAMES_PER_WEATHER:
                    try:
                        cam_img, lidar_pts, radar_pts = setup.tick()
                    except queue.Empty:
                        errors += 1
                        logger.warning("Timeout %d/%d — skipping tick",
                                       errors, MAX_ERRORS)
                        if errors >= MAX_ERRORS:
                            logger.error(
                                "Too many timeouts. Reduce LIDAR_PPS in config.py."
                            )
                            break
                        continue

                    errors = 0

                    spectator_follow(setup.world, setup.vehicle)
                    cam_img   = apply_camera_degradation(cam_img, weather)
                    lidar_pts = apply_lidar_degradation(lidar_pts, weather)
                    labels    = setup.get_bounding_boxes()

                    save_frame(total, weather, cam_img, lidar_pts, radar_pts, labels)

                    collected += 1
                    total     += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        "pts"  : lidar_pts.shape[0],
                        "radar": radar_pts.shape[0],
                        "objs" : len(labels),
                    })

            logger.info("Done: %s  %d/%d frames collected.",
                        weather, collected, FRAMES_PER_WEATHER)

        logger.info("Collection complete. Total frames: %d", total)
        logger.info("Data root: %s", DATA_ROOT)

    finally:
        setup.destroy()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    collect()
