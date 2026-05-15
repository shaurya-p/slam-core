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

## Rerun channels logged

`scripts/rerun_euroc_debug.py` logs the following time-series channels (timeline: `time`, sequence-relative seconds):

| Channel | Unit | Notes |
|---|---|---|
| `imu/gyro/x`, `y`, `z` | rad/s | Raw gyroscope components, body frame |
| `imu/gyro/norm_radps` | rad/s | `‖ω‖` — total angular rate magnitude |
| `imu/accel/x`, `y`, `z` | m/s² | Raw accelerometer components, body frame |
| `imu/accel/norm_mps2` | m/s² | `‖a‖` — total specific force magnitude |
| `imu/accel/norm_minus_gravity_mps2` | m/s² | `‖a‖ − 9.81`; near 0 when static and gravity-aligned |
| `camera/cam0`, `cam1` | image | Grayscale stereo frames |

`norm_minus_gravity_mps2` is a quick sanity signal: a near-zero mean indicates the IMU is gravity-dominated (static or slow motion). Deviations reveal dynamics or calibration offset.

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
