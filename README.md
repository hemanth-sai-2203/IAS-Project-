# RA-ASF — Environment Setup & Run Guide

> **Risk-Aware Adaptive Sensor Fusion for Adverse Weather Autonomous Navigation**
> Member 1 — Simulation & Data Collection
>
> **Windows | CARLA 0.9.15 | Python 3.7 | Intel i7 Iris**

---

## Your Machine Reference

| Item | Value |
|---|---|
| OS | Windows (your current setup) |
| CARLA version | 0.9.15 (already installed) |
| CARLA location | `C:\Users\heman\Downloads\CARLA_0.9.15\WindowsNoEditor` |
| Python version | 3.7 (same as `carla_env37`) |
| Project location | `C:\Users\heman\Music\ra_asf` |
| New environment | `ra_asf_env` (fresh — no conflicts) |

---

## Before You Start — Important Notes

### Why a new environment instead of reusing carla_env37

`carla_env37` works for `rl_imu_project` but has loose version pins
(`torch>=1.9.0`, `numpy>=1.19.0`). Creating `ra_asf_env` fresh with
exact pins ensures Members 2 and 3 get identical results when they run
the same DataLoader code on their machines.

### Why ultralytics (YOLOv8) is NOT in requirements.txt

YOLOv8 requires Python 3.8 or higher. Your CARLA environment is Python 3.7.
This is fine because:
- **Member 1 (you):** only needs data collection — no YOLOv8 required
- **Member 2:** builds their Camera+RADAR model — needs a separate Python 3.8 environment (covered in their own README)
- **Member 3:** same as Member 2

### NUM_WORKERS must be 0 on Windows

Python's multiprocessing DataLoader workers cause a pickle error on Windows
unless the script runs under `if __name__ == '__main__':`.
`config.py` already sets `NUM_WORKERS = 0`. Do not change this.

---

## Step 1 — Place the Project Files

Copy the `ra_asf` folder to:
```
C:\Users\heman\Music\ra_asf\
```

Your folder structure should look exactly like this:
```
C:\Users\heman\Music\ra_asf\
├── config.py
├── requirements.txt
├── verify_setup.py
├── README.md
├── simulation\
│   ├── __init__.py
│   ├── carla_setup.py
│   ├── weather_engine.py
│   └── data_collector.py
└── data\
    ├── __init__.py
    └── dataloader.py
```

---

## Step 2 — Add CARLA Python Module to PYTHONPATH

CARLA ships as a `.egg` file. You need to add it to your environment
so Python can `import carla`.

You already did this for `carla_env37`. We need to do the same for
`ra_asf_env`. The easiest way is to add it as a permanent system variable.

### Check if it is already a system variable

Open Command Prompt and type:
```
echo %PYTHONPATH%
```

If you see the CARLA egg path already listed — skip to Step 3.

### If not already set — add it permanently

1. Press `Win + S`, search for **"Environment Variables"**
2. Click **"Edit the system environment variables"**
3. Click **"Environment Variables..."** button
4. Under **"User variables"**, find `PYTHONPATH`
   - If it exists: click Edit → New → paste the egg path
   - If it does not exist: click New → Variable name: `PYTHONPATH` → Value: the egg path
5. Egg path to add:
```
C:\Users\heman\Downloads\CARLA_0.9.15\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg
```
6. Click OK on all dialogs
7. **Close and reopen** any Command Prompt windows for the change to take effect


cd C:\Users\heman\Music\ra_asf
C:\Users\heman\AppData\Local\Programs\Python\Python37\python.exe -m venv ra_asf_env

ra_asf_env\Scripts\activate

### Verify CARLA is importable (in any Python 3.7 environment)

```
python -c "import carla; print('carla OK')"
```

Expected output: `carla OK`

---

## Step 3 — Create the ra_asf_env Environment

Open **Command Prompt** (not PowerShell — some venv commands behave
differently in PowerShell):

```
cd C:\Users\heman\Music\ra_asf

python -m venv ra_asf_env
```

Activate it:
```
ra_asf_env\Scripts\activate
```

Your prompt should now show `(ra_asf_env)` at the start.

Upgrade pip:
```
python -m pip install --upgrade pip
```

---

## Step 4 — Install Dependencies

### Step 4a — Install PyTorch (CPU version, Python 3.7 compatible)

PyTorch must be installed separately because the CPU wheel URL is different
from the default PyPI index:

```
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

This downloads ~200 MB. Wait for it to complete before proceeding.

Verify:
```
python -c "import torch; print(torch.__version__)"
```
Expected: `1.13.1+cpu`

### Step 4b — Install all other requirements

```
pip install -r requirements.txt
```

### Step 4c — Verify open3d separately (sometimes needs a retry)

```
python -c "import open3d; print(open3d.__version__)"
```

If this fails with a DLL error on Windows, run:
```
pip install open3d==0.15.2 --force-reinstall
```

---

## Step 5 — Run the Verification Script

With `ra_asf_env` still active:

```
cd C:\Users\heman\Music\ra_asf
python verify_setup.py
```

### Expected output (CARLA server NOT running yet)

```
============================================================
  RA-ASF Environment Verification
  Windows | CARLA 0.9.15 | Python 3.7
============================================================

  [PASS]  Python 3.7+   -> Python 3.7.x
  [PASS]  numpy          -> numpy 1.21.x
  [PASS]  opencv-python  -> opencv 4.7.0.72
  [PASS]  PyTorch        -> torch 1.13.1+cpu  [CPU only (Intel Iris)]
  [PASS]  open3d         -> open3d 0.15.2
  [PASS]  tqdm           -> tqdm 4.64.1
  [PASS]  pandas         -> pandas 1.3.x
  [PASS]  scipy          -> scipy 1.7.x
  [PASS]  CARLA .egg file exists
  [PASS]  CARLA Python module importable
  [WARN]  CARLA server not detected.

       Start CARLA first:
       cd C:\Users\heman\Downloads\CARLA_0.9.15\WindowsNoEditor
       CarlaUE4.exe -quality-level=Low -windowed -ResX=800 -ResY=600

       this is alternative command, if the above one gives fatal issue
       CarlaUE4.exe -log
       Wait ~30 seconds, then re-run this script.

  [PASS]  config.py (paths + NUM_WORKERS=0)
  [PASS]  Disk write permission   -> Write OK -> C:\Users\heman\Music\ra_asf\data\collected

============================================================
  Dependencies OK. Start CARLA, then run:
    python simulation\data_collector.py
============================================================
```

All lines must show `[PASS]` before you proceed. The CARLA server WARN is
expected at this stage.

---

## Step 6 — Start CARLA 0.9.15

Open a **second Command Prompt window** (keep your project window open):

```
cd C:\Users\heman\Downloads\CARLA_0.9.15\WindowsNoEditor
```

### Recommended launch command for Intel Iris

```
CarlaUE4.exe -quality-level=Low -windowed -ResX=800 -ResY=600
```

### If CARLA crashes or shows a black window

```
CarlaUE4.exe -quality-level=Low -dx11 -windowed -ResX=800 -ResY=600
```

The `-dx11` flag forces DirectX 11 instead of DirectX 12. Intel Iris
has better compatibility with DX11.

### If it still crashes — minimal mode

```
CarlaUE4.exe -quality-level=Low -windowed -ResX=640 -ResY=480 -benchmark -fps=20
```

Wait until you see in the CARLA window or console:
```
LogCarla: Initialized GameInstance
```
This takes **30–60 seconds** on first load.

---

## Step 7 — Re-run Verification with CARLA Running

Back in your project window (with `ra_asf_env` active):

```
python verify_setup.py
```

Now you should see:
```
  [PASS]  CARLA server running   -> CARLA server v0.9.15  at localhost:2000
  ...
  ALL CHECKS PASSED. Ready to collect data.
```

---

## Step 8 — Collect Data

```
cd C:\Users\heman\Music\ra_asf
ra_asf_env\Scripts\activate
python simulation\data_collector.py
```

### What you will see in the console

```
============================================================
Weather: clear         target: 500 frames
============================================================
Warming up (30 ticks) ...
clear:     100%|████████| 500/500 [pts=58234, radar=6, objs=9]

============================================================
Weather: fog_light     target: 500 frames
============================================================
fog_light: 100%|████████| 500/500 [pts=43100, radar=5, objs=8]

============================================================
Weather: fog_heavy     target: 500 frames
============================================================
fog_heavy: 100%|████████| 500/500 [pts=16800, radar=5, objs=7]

============================================================
Weather: rain          target: 500 frames
============================================================
rain:      100%|████████| 500/500 [pts=46200, radar=6, objs=8]

Collection complete. Total frames: 2000
```

### Expected LiDAR point counts

| Weather | Expected pts (approx) |
|---|---|
| clear | 55,000 – 65,000 |
| fog_light | 35,000 – 50,000 |
| fog_heavy | 10,000 – 22,000 |
| rain | 40,000 – 55,000 |

If `clear` gives fewer than 40,000 points your CARLA is running too slowly.
See Troubleshooting below.

---

## Step 9 — Verify the Dataset

```
python data\dataloader.py
```

Expected output:
```
── Batch shapes ───────────────────────────
  image   : torch.Size([4, 3, 600, 800])
  lidar   : torch.Size([4, 60000, 4])
  radar   : torch.Size([4, 64, 4])
  weather : ['clear', 'fog_heavy', 'rain', 'fog_light']
  n_objs  : [9, 5, 8, 7]
```

If this runs without error, Member 1's work is complete.

---

## Output Directory Structure

After collection, your data folder looks like this:
```
C:\Users\heman\Music\ra_asf\data\collected\
├── clear\
│   ├── images\    frame_000000.jpg  ...  frame_000499.jpg
│   ├── lidar\     frame_000000.npy  ...  frame_000499.npy
│   ├── radar\     frame_000000.npy  ...  frame_000499.npy
│   └── labels\    frame_000000.json ...  frame_000499.json
├── fog_light\
├── fog_heavy\
└── rain\
```

**Share the entire `data\collected\` folder with Members 2 and 3.**
They do not need CARLA installed. They only need this folder and
`data\dataloader.py`.

---

## Troubleshooting

### "Connection refused" on port 2000
CARLA is not running or still loading. Wait and retry. Check that no
other program is using port 2000.

### "No spawn points found"
Town03 did not load. Restart CARLA completely and wait the full 60 seconds.

### Sensor timeout warnings during collection
Your machine cannot keep up with current settings. Open `config.py` and:
```python
LIDAR_PPS           = 500_000     # reduce from 700_000
FIXED_DELTA_SECONDS = 0.1         # reduce to 10 Hz
QUEUE_TIMEOUT       = 12.0        # increase wait time
```

### CARLA black screen / crash on startup
Try `-dx11` flag. Intel Iris works better with DirectX 11 than 12.

### "carla module not found" after setting PYTHONPATH
Close all Command Prompt windows and open a new one. Windows environment
variables only apply to newly opened terminals.

### open3d import fails with DLL error
```
pip install open3d==0.15.2 --force-reinstall
```
Also install Visual C++ Redistributable if missing:
https://aka.ms/vs/17/release/vc_redist.x64.exe

### PyTorch import fails
Make sure you installed the `+cpu` version:
```
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```
Do NOT use the standard `pip install torch` — that tries to install the
CUDA version which will fail on Intel Iris.

### DataLoader hangs forever
This means `NUM_WORKERS` is not 0. Check `config.py`:
```python
NUM_WORKERS = 0    # must be 0 on Windows
```

---

## Disk Usage Estimate

| Content | Size |
|---|---|
| 500 JPEG images × 4 weathers | ~800 MB |
| 500 LiDAR .npy × 4 weathers  | ~2.5 GB |
| 500 RADAR .npy × 4 weathers  | ~5 MB |
| 500 label .json × 4 weathers | ~20 MB |
| **Total** | **~3.5 GB** |

Ensure you have at least **5 GB free** before starting.

---

## Quick Command Reference

```
# Activate environment
C:\Users\heman\Music\ra_asf> ra_asf_env\Scripts\activate

# Run verification
(ra_asf_env) C:\Users\heman\Music\ra_asf> python verify_setup.py

# Collect data (CARLA must be running first)
(ra_asf_env) C:\Users\heman\Music\ra_asf> python simulation\data_collector.py

# Verify dataset
(ra_asf_env) C:\Users\heman\Music\ra_asf> python data\dataloader.py
```

---

## Team Reference

| Member | What they need from you |
|---|---|
| Member 2 | Copy of `data\collected\` folder + `data\dataloader.py` |
| Member 3 | Copy of `data\collected\` folder + `data\dataloader.py` |

Both Members 2 and 3 need Python 3.8 for YOLOv8. Their setup is
separate from this environment.







How to run data code the sensor health file 
Open your Command Prompt.

Activate your virtual environment just like you usually do (ra_asf_env).

Navigate to your main project folder (not the simulation folder):
cd C:\Users\heman\Music\ra_asf

Run the script:
python simulation\sensor_health_monitor.py

What You Will See
When you hit enter, it will immediately print out the results of the three tests you coded:

Test 1: It will show high scores (near 1.0) and suggest the "GOLD" mode because it fed the system perfect, clear-weather dummy arrays.

Test 2: It will show the Camera and LiDAR scores plummeting, and the system should suggest "M1" (or whatever your RADAR fallback is) because it applied a heavy Gaussian blur to the dummy image.

Test 3: It will show the Camera score hit 0.0 (because it passed a pitch-black array) and suggest "M3" (LiDAR + RADAR).



1. Navigate to your project folder
Open your Command Prompt and type:
cd C:\Users\heman\Music\ra_asf

2. Activate the virtual environment
Since you are on Windows, type this exact command and hit Enter:
ra_asf_env\Scripts\activate
(You will know it worked when you see (ra_asf_env) appear at the very beginning of your command line).

3. Run the script
Now that the environment is awake and knows where all your libraries are, run your test:
python simulation\sensor_health_monitor.py

(And you can test the selector right after by running python simulation\fusion_selector.py)
