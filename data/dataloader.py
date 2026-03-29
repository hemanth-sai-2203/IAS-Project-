"""
dataloader.py
-------------
PyTorch Dataset + DataLoader for the RA-ASF multi-weather sensor dataset.

Windows critical note:
    NUM_WORKERS must be 0 on Windows.
    Python multiprocessing with DataLoader workers causes a pickle error
    on Windows when workers > 0 and the script is not under
    `if __name__ == '__main__':` guard.
    config.py already sets NUM_WORKERS = 0.

Usage:
    from data.dataloader import build_dataloaders
    train_loader, val_loader = build_dataloaders()
    for batch in train_loader:
        img    = batch["image"]    # (B, 3, H, W)  float32 [0,1]
        lidar  = batch["lidar"]    # (B, 80000, 4) float32
        radar  = batch["radar"]    # (B, 64, 4)    float32
        labels = batch["labels"]   # list of dicts
"""

import os
import sys
import json
import glob
import logging
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import (
    DATA_ROOT, WEATHER_STATES,
    CAMERA_WIDTH, CAMERA_HEIGHT,
    TRAIN_VAL_SPLIT, BATCH_SIZE, NUM_WORKERS,
)

logger = logging.getLogger(__name__)

LIDAR_MAX_POINTS = 60_000   # reduced from 80k to match lower PPS on this machine
RADAR_MAX_POINTS = 64


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _pad_or_clip(arr, max_pts):
    """Ensure array has exactly max_pts rows via zero-pad or random sub-sample."""
    n    = arr.shape[0]
    cols = arr.shape[1] if arr.ndim == 2 else 1

    if n == 0:
        return np.zeros((max_pts, cols), dtype=np.float32)
    if n >= max_pts:
        idx = np.random.choice(n, max_pts, replace=False)
        return arr[idx].astype(np.float32)
    pad = np.zeros((max_pts - n, cols), dtype=np.float32)
    return np.vstack([arr.astype(np.float32), pad])


def _normalise_image(img):
    """uint8 (H,W,3) → float32 (3,H,W) in [0,1], resized if needed."""
    if img.shape[:2] != (CAMERA_HEIGHT, CAMERA_WIDTH):
        img = cv2.resize(img, (CAMERA_WIDTH, CAMERA_HEIGHT))
    img = img.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))     # HWC → CHW


# ── DATASET ───────────────────────────────────────────────────────────────────

class RaAsfDataset(Dataset):
    def __init__(self, data_root=DATA_ROOT, weathers=None,
                 max_lidar=LIDAR_MAX_POINTS, max_radar=RADAR_MAX_POINTS):
        self.data_root = data_root
        self.weathers  = weathers or WEATHER_STATES
        self.max_lidar = max_lidar
        self.max_radar = max_radar
        self.samples   = self._index()

        if len(self.samples) == 0:
            raise RuntimeError(
                "No samples found in {}.\n"
                "Run data collection first:\n"
                "  python simulation\\data_collector.py".format(data_root)
            )
        logger.info("Dataset: %d samples  weathers=%s",
                    len(self.samples), self.weathers)

    def _index(self):
        samples = []
        for weather in self.weathers:
            img_dir = os.path.join(self.data_root, weather, "images")
            if not os.path.isdir(img_dir):
                logger.warning("Missing: %s", img_dir)
                continue
            for img_path in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
                stem = os.path.splitext(os.path.basename(img_path))[0]
                entry = {
                    "weather"   : weather,
                    "img_path"  : img_path,
                    "lidar_path": os.path.join(self.data_root, weather, "lidar",  stem + ".npy"),
                    "radar_path": os.path.join(self.data_root, weather, "radar",  stem + ".npy"),
                    "label_path": os.path.join(self.data_root, weather, "labels", stem + ".json"),
                }
                if all(os.path.exists(v) for k, v in entry.items() if k != "weather"):
                    samples.append(entry)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        e = self.samples[idx]

        img_bgr = cv2.imread(e["img_path"])
        if img_bgr is None:
            raise IOError("Cannot read: {}".format(e["img_path"]))
        img_t = torch.from_numpy(
            _normalise_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        )

        lidar_t = torch.from_numpy(
            _pad_or_clip(np.load(e["lidar_path"]), self.max_lidar)
        )
        radar_t = torch.from_numpy(
            _pad_or_clip(np.load(e["radar_path"]), self.max_radar)
        )

        with open(e["label_path"]) as f:
            ldata = json.load(f)

        return {
            "image"   : img_t,
            "lidar"   : lidar_t,
            "radar"   : radar_t,
            "labels"  : ldata["objects"],
            "weather" : e["weather"],
            "frame_id": ldata.get("frame_id", idx),
        }


# ── COLLATE ───────────────────────────────────────────────────────────────────

def collate_fn(batch):
    return {
        "image"   : torch.stack([b["image"]  for b in batch]),
        "lidar"   : torch.stack([b["lidar"]  for b in batch]),
        "radar"   : torch.stack([b["radar"]  for b in batch]),
        "labels"  : [b["labels"]   for b in batch],
        "weather" : [b["weather"]  for b in batch],
        "frame_id": [b["frame_id"] for b in batch],
    }


# ── BUILD LOADERS ─────────────────────────────────────────────────────────────

def build_dataloaders(data_root=DATA_ROOT, weathers=None,
                      batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                      train_split=TRAIN_VAL_SPLIT, seed=42):
    """
    Returns (train_loader, val_loader).
    num_workers is forced to 0 on Windows.
    """
    if os.name == "nt" and num_workers != 0:
        logger.warning("Windows detected — forcing num_workers=0.")
        num_workers = 0

    ds      = RaAsfDataset(data_root=data_root, weathers=weathers)
    n_train = int(len(ds) * train_split)
    n_val   = len(ds) - n_train
    gen     = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)

    kw = dict(collate_fn=collate_fn, pin_memory=False, num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, drop_last=False, **kw)

    logger.info("DataLoaders: train=%d  val=%d", len(train_ds), len(val_ds))
    return train_loader, val_loader


# ── SANITY CHECK ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s")
    train_loader, val_loader = build_dataloaders()
    batch = next(iter(train_loader))
    print("\n── Batch shapes ───────────────────────────")
    print("  image   :", batch["image"].shape)
    print("  lidar   :", batch["lidar"].shape)
    print("  radar   :", batch["radar"].shape)
    print("  weather :", batch["weather"])
    print("  n_objs  :", [len(l) for l in batch["labels"]])
