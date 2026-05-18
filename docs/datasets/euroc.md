# EuRoC Dataset Setup

## Dataset root resolution

Scripts resolve the dataset root in this priority order:

1. **CLI argument** — `--dataset-root /path/to/datasets`
2. **Environment variable** — `SLAM_CORE_DATASETS=/path/to/datasets`
3. **Config field** — `dataset_root: /path/to/datasets` in the sequence yaml
4. **Error** — if none of the above are set, the script exits with a message listing all three options

The committed configs contain no `dataset_root` field so they stay portable across machines.

Recommended setup:

```bash
export SLAM_CORE_DATASETS=~/datasets
```

## Expected layout

```
<dataset_root>/
  euroc/
    MH_01_easy/
      mav0/
        imu0/data.csv
        cam0/data.csv
        cam0/data/   ← image files
        cam1/data.csv
        cam1/data/   ← image files
    MH_02_easy/
    V1_01_easy/
    ...
```

EuRoC sequences are never committed to this repo. Download from:
https://rpg.ifi.uzh.ch/docs/IJRR17_Burri.html

## IMU CSV format

```
timestamp [ns], w_RS_S_x [rad s^-1], w_RS_S_y [rad s^-1], w_RS_S_z [rad s^-1],
a_RS_S_x [m s^-2], a_RS_S_y [m s^-2], a_RS_S_z [m s^-2]
```

Timestamps are in nanoseconds. Scripts convert to seconds internally.

## Camera CSV format

```
timestamp [ns], filename
```

Images are 752×480 grayscale PNG. Stereo baseline ~11 cm, left/right synchronized.

## Rerun channels — debug script

`scripts/rerun_euroc_debug.py` logs the following channels (timeline: `time`, sequence-relative seconds):

| Entity path | Unit | Notes |
|---|---|---|
| `imu/gyro/x`, `y`, `z` | rad/s | Raw gyroscope components, body frame |
| `imu/accel/x`, `y`, `z` | m/s² | Raw accelerometer components, body frame |
| `imu_derived/gyro_norm_radps` | rad/s | `‖ω‖` — total angular rate magnitude |
| `imu_derived/accel_norm_mps2` | m/s² | `‖a‖` — total specific force magnitude |
| `imu_derived/accel_norm_minus_gravity_mps2` | m/s² | `‖a‖ − 9.81`; near 0 when static and gravity-aligned |
| `camera/cam0`, `cam1` | image | Grayscale stereo frames |

Raw channels are logged under `imu/`. Derived channels are under `imu_derived/` as a separate top-level entity so they appear as a distinct group in the Rerun tree.

`accel_norm_minus_gravity_mps2` near zero indicates the IMU is gravity-dominated (static or slow motion). Deviations reveal dynamics or calibration offset.

## Rerun channels — gyro propagation visualization

First export the orientation CSV from the C++ tool:

```bash
./build/tools/export_gyro_propagation \
  "$HOME/datasets/euroc/MH_01_easy/mav0/imu0/data.csv" \
  results/imu/MH_01_easy_gyro_orientations.csv
```

Then run the visualization:

```bash
uv run python scripts/rerun_euroc_gyro_propagation.py \
  configs/datasets/euroc_mh01.yaml \
  results/imu/MH_01_easy_gyro_orientations.csv \
  --dataset-root "$HOME/datasets" \
  --imu-stride 1 --frame-stride 5 --max-duration-s 30
# Output: results/rerun/MH_01_easy_gyro_propagation.rrd
```

`scripts/rerun_euroc_gyro_propagation.py` logs the following channels:

| Entity path | Unit | Notes |
|---|---|---|
| `orientation/body_frame` | — | 3D arrows: estimated x-axis (red, full length), y/z (gray, shorter) |
| `orientation/body_frame_gt` | — | 3D arrows: GT x-axis (blue, full length), y/z (gray, shorter) |
| `orientation_error/geodesic_deg` | deg | SO(3) geodesic angle between estimated and GT rotation |
| `orientation_error/roll_deg` | deg | ZYX roll error component |
| `orientation_error/pitch_deg` | deg | ZYX pitch error component |
| `orientation_error/yaw_deg` | deg | ZYX yaw error component |

All orientations and errors are relative to the first estimated/GT pair as reference (both frames coincide at t=0). GT is from `mav0/state_groundtruth_estimate0/data.csv`, matched by nearest-neighbor timestamp lookup. Gyro-only propagation has no gravity alignment, no accelerometer fusion, and no bias correction — drift is expected.

## Rerun blueprint workflow

Rerun separates **data** from **layout**:

- `.rrd` — the recording file. Contains all logged data (images, IMU scalars). Large; gitignored under `results/`.
- `.rbl` — the blueprint file. Contains only the viewer panel layout. Small; committed under `configs/rerun/`.

Committing the `.rbl` makes the dashboard reproducible across recordings and machines without re-arranging the viewer each time.

### Step 1 — generate a debug recording

```bash
uv run python scripts/rerun_euroc_debug.py configs/datasets/euroc_mh01.yaml \
  --dataset-root "$HOME/datasets" \
  --max-frames 300 --image-stride 2 --imu-stride 2
# Output: results/rerun/MH_01_easy_debug.rrd
```

### Step 2 — create the blueprint (one-time, manual)

Open the recording in Rerun:

```bash
uv run rerun results/rerun/MH_01_easy_debug.rrd
```

Arrange the viewport into the following layout:

```
┌──────────────────────────────────────┐
│  camera/cam0      camera/cam1        │  ← stereo images
├──────────────────────────────────────┤
│  imu_derived/gyro_norm_radps         │
│  imu_derived/accel_norm_mps2         │  ← derived magnitudes
│  imu_derived/accel_norm_minus_gravity│
├───────────────────┬──────────────────┤
│  imu/gyro/x       │  imu/accel/x     │
│  imu/gyro/y       │  imu/accel/y     │  ← raw axes
│  imu/gyro/z       │  imu/accel/z     │
└───────────────────┴──────────────────┘
```

Save the layout via **File → Save blueprint** and write it to:

```
configs/rerun/euroc_imu_debug.rbl
```

Commit the `.rbl` file. It does not contain dataset images or IMU data — only panel positions and entity selections.

### Step 3 — open any recording with the blueprint

```bash
uv run rerun results/rerun/MH_01_easy_debug.rrd configs/rerun/euroc_imu_debug.rbl
```

The blueprint applies to any `.rrd` that logs the same entity paths, so the same layout works across sequences and future recordings.

## Configs

Each sequence has a config under `configs/datasets/`:

```
configs/datasets/euroc_mh01.yaml
```

Pass the config path to scripts:

```bash
# Using env var (recommended)
export SLAM_CORE_DATASETS=~/datasets
uv run python scripts/rerun_euroc_debug.py configs/datasets/euroc_mh01.yaml

# Using CLI arg (overrides env var)
uv run python scripts/rerun_euroc_debug.py configs/datasets/euroc_mh01.yaml --dataset-root ~/datasets
```
