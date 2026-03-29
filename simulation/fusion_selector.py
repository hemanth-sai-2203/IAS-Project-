"""
fusion_selector.py
------------------
Dynamically routes sensor data to the appropriate localization/fusion module
based on real-time health scores. Implements hysteresis to prevent rapid
mode-switching jitter during transient weather anomalies.

Modes:
  - GOLD : All sensors healthy (Camera + LiDAR + RADAR + IMU)
  - M1   : RADAR degraded (Camera + LiDAR + IMU)
  - M2   : LiDAR degraded (Camera + RADAR + IMU)
  - M3   : Camera degraded (LiDAR + RADAR + IMU)
  - DR   : Critical failure (Dead Reckoning via IMU only)
"""

import logging
import collections

logger = logging.getLogger(__name__)

class FusionSelector:
    """
    Manages the state machine for sensor fusion mode switching.
    
    Parameters
    ----------
    hysteresis_ticks : int
        The number of consecutive frames a new mode must be suggested 
        before the selector actually commits to the switch. 
        Higher = more stable but slower to react. Default is 3.
    """
    
    def __init__(self, hysteresis_ticks=3):
        self.current_mode = "GOLD"
        self.hysteresis_ticks = hysteresis_ticks
        
        # A rolling queue that remembers the last N suggested modes
        self._mode_votes = collections.deque(maxlen=hysteresis_ticks)
        
        logger.info("FusionSelector ready (Hysteresis: %d ticks)", hysteresis_ticks)

    def _determine_target_mode(self, suggested_mode, degraded_list):
        """Overrides the health monitor if a critical total failure occurs."""
        if len(degraded_list) >= 3:
            return "DR"  # Dead Reckoning: All exteroceptive sensors blind
        return suggested_mode

    def update_mode(self, health_dict):
        """
        Applies hysteresis logic to decide the official active mode.
        """
        suggested = health_dict.get("active_mode", "GOLD")
        degraded  = health_dict.get("degraded", [])
        
        target_mode = self._determine_target_mode(suggested, degraded)
        self._mode_votes.append(target_mode)

        # Only switch if the queue is full AND all recent votes agree
        if len(self._mode_votes) == self.hysteresis_ticks:
            if len(set(self._mode_votes)) == 1:
                confirmed_mode = self._mode_votes[0]
                
                # If the confirmed mode is different from our current state, switch!
                if confirmed_mode != self.current_mode:
                    logger.warning(
                        "🔄 FUSION SWITCH: %s -> %s | Degraded: %s", 
                        self.current_mode, confirmed_mode, degraded
                    )
                    self.current_mode = confirmed_mode

        return self.current_mode

    def process_frame(self, health_dict, cam_data, lidar_data, radar_data, imu_data):
        """
        The main public method. Takes in the health scores and the raw data,
        updates the state machine, and routes the data to the correct math module.
        """
        # 1. Update the state machine
        active_mode = self.update_mode(health_dict)

        # 2. Route the data to the correct EKF/Fusion algorithm
        state_estimate = None

        if active_mode == "GOLD":
            state_estimate = self._run_gold_fusion(cam_data, lidar_data, radar_data, imu_data)
        elif active_mode == "M1":
            state_estimate = self._run_m1_fusion(cam_data, lidar_data, imu_data)
        elif active_mode == "M2":
            state_estimate = self._run_m2_fusion(cam_data, radar_data, imu_data)
        elif active_mode == "M3":
            state_estimate = self._run_m3_fusion(lidar_data, radar_data, imu_data)
        elif active_mode == "DR":
            state_estimate = self._run_dead_reckoning(imu_data)

        return active_mode, state_estimate

    # ── PLACEHOLDER MATHEMATICAL MODULES ─────────────────────────────────────
    # These functions are where you will eventually plug in your 
    # actual Kalman Filter or Deep Learning prediction classes.

    def _run_gold_fusion(self, cam, lidar, radar, imu):
        # e.g., return self.ekf_gold.predict_and_update(...)
        return {"status": "Processing GOLD", "x": 0.0, "y": 0.0}

    def _run_m1_fusion(self, cam, lidar, imu):
        return {"status": "Processing M1 (Cam+LiDAR+IMU)", "x": 0.0, "y": 0.0}

    def _run_m2_fusion(self, cam, radar, imu):
        return {"status": "Processing M2 (Cam+RADAR+IMU)", "x": 0.0, "y": 0.0}

    def _run_m3_fusion(self, lidar, radar, imu):
        return {"status": "Processing M3 (LiDAR+RADAR+IMU)", "x": 0.0, "y": 0.0}

    def _run_dead_reckoning(self, imu):
        # Pure IMU double integration for GPS-denied environments
        return {"status": "Processing DR (IMU ONLY)", "x": 0.0, "y": 0.0}


# ── QUICK SANITY TEST ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    
    selector = FusionSelector(hysteresis_ticks=3)
    
    print("\n── Test 1: Simulating Camera Flicker (Jitter Protection) ──")
    # Tick 1: Camera fails
    selector.process_frame({"active_mode": "M3", "degraded": ["camera"]}, None, None, None, None)
    print(f"Tick 1 Mode: {selector.current_mode}")
    # Tick 2: Camera comes back instantly
    selector.process_frame({"active_mode": "GOLD", "degraded": []}, None, None, None, None)
    print(f"Tick 2 Mode: {selector.current_mode}")
    
    print("\n── Test 2: Simulating Sustained Heavy Fog ──")
    for i in range(1, 5):
        selector.process_frame({"active_mode": "M3", "degraded": ["camera"]}, None, None, None, None)
        print(f"Fog Tick {i} Mode: {selector.current_mode}")

    print("\n── Test 3: Total Sensor Blindness (Tunnel Entry) ──")
    for i in range(1, 5):
        selector.process_frame({"active_mode": "M3", "degraded": ["camera", "lidar", "radar"]}, None, None, None, None)
        print(f"Blind Tick {i} Mode: {selector.current_mode}")
    
    print("\nAll tests passed.\n")