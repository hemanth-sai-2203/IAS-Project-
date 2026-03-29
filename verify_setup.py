"""
verify_setup.py
---------------
Run this before anything else.
Checks every dependency and CARLA connectivity.

Usage (from project root):
    cd C:\\Users\\heman\\Music\\ra_asf
    ra_asf_env\\Scripts\\activate
    python verify_setup.py
"""

import sys
import os

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"


def check(label, fn):
    try:
        result = fn()
        suffix = "   -> " + result if result else ""
        print("{}  {}{}".format(PASS, label, suffix))
        return True
    except Exception as exc:
        print("{}  {}".format(FAIL, label))
        print("         {}".format(exc))
        return False


print()
print("=" * 60)
print("  RA-ASF Environment Verification")
print("  Windows | CARLA 0.9.15 | Python 3.7")
print("=" * 60)
print()

all_ok = True

# ── Python ────────────────────────────────────────────────────────────────────
def chk_python():
    v = sys.version_info
    if v.major != 3 or v.minor < 7:
        raise EnvironmentError("Python 3.7+ required, got {}.{}".format(v.major, v.minor))
    return "Python {}.{}.{}".format(v.major, v.minor, v.micro)
all_ok &= check("Python 3.7+", chk_python)

# ── numpy ─────────────────────────────────────────────────────────────────────
def chk_numpy():
    import numpy as np
    return "numpy " + np.__version__
all_ok &= check("numpy", chk_numpy)

# ── opencv ────────────────────────────────────────────────────────────────────
def chk_cv2():
    import cv2
    return "opencv " + cv2.__version__
all_ok &= check("opencv-python", chk_cv2)

# ── PyTorch ───────────────────────────────────────────────────────────────────
def chk_torch():
    import torch
    cuda = "CUDA available" if torch.cuda.is_available() else "CPU only (Intel Iris — expected)"
    return "torch {}  [{}]".format(torch.__version__, cuda)
all_ok &= check("PyTorch", chk_torch)

# ── open3d ────────────────────────────────────────────────────────────────────
def chk_open3d():
    import open3d as o3d
    return "open3d " + o3d.__version__
all_ok &= check("open3d", chk_open3d)

# ── tqdm ─────────────────────────────────────────────────────────────────────
def chk_tqdm():
    import tqdm
    return "tqdm " + tqdm.__version__
all_ok &= check("tqdm", chk_tqdm)

# ── pandas ───────────────────────────────────────────────────────────────────
def chk_pandas():
    import pandas as pd
    return "pandas " + pd.__version__
all_ok &= check("pandas", chk_pandas)

# ── scipy ─────────────────────────────────────────────────────────────────────
def chk_scipy():
    import scipy
    return "scipy " + scipy.__version__
all_ok &= check("scipy", chk_scipy)

# ── CARLA egg path ────────────────────────────────────────────────────────────
def chk_carla_egg():
    from config import CARLA_EGG
    if not os.path.exists(CARLA_EGG):
        raise FileNotFoundError(
            "Egg not found: {}\n"
            "         Check CARLA_ROOT in config.py".format(CARLA_EGG)
        )
    return CARLA_EGG
egg_ok = check("CARLA .egg file exists", chk_carla_egg)
all_ok &= egg_ok

# ── CARLA importable ──────────────────────────────────────────────────────────
def chk_carla_import():
    import carla
    return "carla module imported OK"
carla_ok = check("CARLA Python module importable", chk_carla_import)
all_ok &= carla_ok

# ── CARLA server ─────────────────────────────────────────────────────────────
server_ok = False
if carla_ok:
    def chk_server():
        import carla
        from config import CARLA_HOST, CARLA_PORT
        c = carla.Client(CARLA_HOST, CARLA_PORT)
        c.set_timeout(5.0)
        v = c.get_server_version()
        return "CARLA server v{}  at {}:{}".format(v, CARLA_HOST, CARLA_PORT)
    server_ok = check("CARLA server running", chk_server)
    if not server_ok:
        print()
        print("{}  Start CARLA first:".format(WARN))
        print("       cd C:\\Users\\heman\\Downloads\\CARLA_0.9.15\\WindowsNoEditor")
        print("       CarlaUE4.exe -quality-level=Low -windowed -ResX=800 -ResY=600")
        print("       Wait ~30 seconds, then re-run this script.")
        print()

# ── config.py ─────────────────────────────────────────────────────────────────
def chk_config():
    from config import DATA_ROOT, WEATHER_STATES, FRAMES_PER_WEATHER, NUM_WORKERS
    if NUM_WORKERS != 0:
        raise EnvironmentError(
            "NUM_WORKERS={} — must be 0 on Windows.".format(NUM_WORKERS)
        )
    return "DATA_ROOT={} | {} frames/weather".format(DATA_ROOT, FRAMES_PER_WEATHER)
all_ok &= check("config.py (paths + NUM_WORKERS=0)", chk_config)

# ── disk write ────────────────────────────────────────────────────────────────
def chk_write():
    from config import DATA_ROOT
    os.makedirs(DATA_ROOT, exist_ok=True)
    test = os.path.join(DATA_ROOT, ".write_test")
    with open(test, "w") as f:
        f.write("ok")
    os.remove(test)
    return "Write OK -> " + DATA_ROOT
all_ok &= check("Disk write permission", chk_write)

# ── summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
if all_ok and server_ok:
    print("  ALL CHECKS PASSED. Ready to collect data.")
    print()
    print("  Run:")
    print("    python simulation\\data_collector.py")
elif all_ok and not server_ok:
    print("  Dependencies OK. Start CARLA, then run:")
    print("    python simulation\\data_collector.py")
else:
    print("  SOME CHECKS FAILED. Fix the issues above first.")
print("=" * 60)
print()
