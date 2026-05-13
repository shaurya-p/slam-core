# EuRoC Dataset Setup

## Environment variable

All external datasets live under a single root directory controlled by:

```
export SLAM_CORE_DATASETS=/path/to/your/datasets
```

## Expected layout

```
$SLAM_CORE_DATASETS/
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

Timestamps are in nanoseconds. The script converts to seconds internally.

## Camera CSV format

```
timestamp [ns], filename
```

Images are 752×480 grayscale PNG. Stereo baseline ~11 cm, left/right synchronized.

## Configs

Each sequence has a config under `configs/datasets/`:

```
configs/datasets/euroc_mh01.yaml
```

Pass the config path to scripts:

```bash
uv run python scripts/rerun_euroc_debug.py configs/datasets/euroc_mh01.yaml
```
