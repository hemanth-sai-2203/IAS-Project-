"""
config.py
---------
Central configuration for RA-ASF project.
All tunable parameters live here.

Paths are set for:
  Windows machine
  CARLA 0.9.15  at  C:\\Users\\heman\\Downloads\\CARLA_0.9.15\\WindowsNoEditor
  Project root  at  C:\\Users\\heman\\Music\\ra_asf
"""

import os

# ── CARLA CONNECTION ──────────────────────────────────────────────────────────
CARLA_HOST        = "localhost"
CARLA_PORT        = 2000
CARLA_TIMEOUT     = 30.0           # seconds to wait for server response

# ── CARLA INSTALL PATH (used by verify_setup.py to locate the .egg) ──────────
CARLA_ROOT = r"C:\Users\heman\Downloads\CARLA_0.9.15\WindowsNoEditor"
CARLA_EGG  = os.path.join(
    CARLA_ROOT,
    "PythonAPI", "carla", "dist",
    "carla-0.9.15-py3.7-win-amd64.egg",
)

# ── PROJECT PATHS ─────────────────────────────────────────────────────────────
PROJECT_ROOT = r"C:\Users\heman\Music\ra_asf"
DATA_ROOT    = os.path.join(PROJECT_ROOT, "data", "collected")

# ── SIMULATION ────────────────────────────────────────────────────────────────
TOWN                 = "Town02"     # suburban map — good variety of scenarios
FIXED_DELTA_SECONDS  = 0.05        # 20 Hz simulation tick
SEED                 = 42

# ── CAMERA SENSOR ─────────────────────────────────────────────────────────────
CAMERA_WIDTH   = 800
CAMERA_HEIGHT  = 600
CAMERA_FOV     = 90
CAMERA_FPS     = 20

# ── LIDAR SENSOR ──────────────────────────────────────────────────────────────
LIDAR_CHANNELS       = 64
LIDAR_RANGE          = 100.0       # metres
LIDAR_PPS            = 700_000     # points per second — reduced for Intel Iris
LIDAR_ROTATION_HZ    = 20
LIDAR_UPPER_FOV      = 10.0
LIDAR_LOWER_FOV      = -30.0

# ── RADAR SENSOR ──────────────────────────────────────────────────────────────
RADAR_HFOV    = 60.0               # horizontal FOV degrees
RADAR_VFOV    = 10.0               # vertical FOV degrees
RADAR_RANGE   = 100.0              # metres
RADAR_PPS     = 4000

# ── SENSOR MOUNT POSITIONS (x, y, z, pitch, yaw, roll) relative to vehicle ───
CAMERA_MOUNT  = (1.5,  0.0, 2.4,  0.0, 0.0, 0.0)
LIDAR_MOUNT   = (0.0,  0.0, 2.8,  0.0, 0.0, 0.0)
RADAR_MOUNT   = (2.0,  0.0, 1.0,  5.0, 0.0, 0.0)

# ── DATA COLLECTION ───────────────────────────────────────────────────────────
WEATHER_STATES      = ["clear", "fog_light", "fog_heavy", "rain"]
FRAMES_PER_WEATHER  = 500          # frames to collect per weather state
QUEUE_TIMEOUT       = 8.0          # seconds — increased for Windows timing

# ── WEATHER PARAMETERS (CARLA 0.9.15 WeatherParameters fields) ───────────────
WEATHER_PARAMS = {
    "clear": {
        "cloudiness":             0.0,
        "precipitation":          0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity":         5.0,
        "sun_azimuth_angle":      70.0,
        "sun_altitude_angle":     70.0,
        "fog_density":            0.0,
        "fog_distance":           0.0,
        "wetness":                0.0,
    },
    "fog_light": {
        "cloudiness":             80.0,
        "precipitation":          0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity":         10.0,
        "sun_azimuth_angle":      70.0,
        "sun_altitude_angle":     50.0,
        "fog_density":            30.0,
        "fog_distance":           30.0,
        "wetness":                0.0,
    },
    "fog_heavy": {
        "cloudiness":             100.0,
        "precipitation":          0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity":         20.0,
        "sun_azimuth_angle":      70.0,
        "sun_altitude_angle":     40.0,
        "fog_density":            80.0,
        "fog_distance":           10.0,
        "wetness":                20.0,
    },
    "rain": {
        "cloudiness":             90.0,
        "precipitation":          80.0,
        "precipitation_deposits": 60.0,
        "wind_intensity":         50.0,
        "sun_azimuth_angle":      70.0,
        "sun_altitude_angle":     30.0,
        "fog_density":            20.0,
        "fog_distance":           50.0,
        "wetness":                90.0,
    },
}

# ── SENSOR DEGRADATION (post-processing — applied after CARLA capture) ────────
DEGRADATION = {
    "clear":     {"blur_sigma": 0.0,  "lidar_dropout": 0.00, "lidar_intensity_scale": 1.0},
    "fog_light": {"blur_sigma": 0.8,  "lidar_dropout": 0.15, "lidar_intensity_scale": 0.7},
    "fog_heavy": {"blur_sigma": 2.0,  "lidar_dropout": 0.55, "lidar_intensity_scale": 0.3},
    "rain":      {"blur_sigma": 2.5,  "lidar_dropout": 0.20, "lidar_intensity_scale": 0.8},
}

# ── DATALOADER ────────────────────────────────────────────────────────────────
TRAIN_VAL_SPLIT  = 0.8
BATCH_SIZE       = 4               # low — no GPU
NUM_WORKERS      = 0               # MUST be 0 on Windows (multiprocessing pickle issue)

# ── HEALTH MONITOR THRESHOLDS ─────────────────────────────────────────────────
LIDAR_EXPECTED_POINTS     = 60_000     # reduced from 70k — accounts for lower PPS
LIDAR_EXPECTED_INTENSITY  = 0.6
RADAR_EXPECTED_DETECTIONS = 8
RADAR_MAX_RCS             = 15.0
BLUR_LAPLACIAN_CLEAR      = 500.0
HEALTH_THRESHOLD          = 0.40
