"""
Plot accelerometer bias evaluation results from evaluate_accel_bias_from_gt.

Three subplots:
  1. Bias components x/y/z over time
  2. Bias norm over time
  3. Velocity error norm over time

Usage:
    uv run python scripts/plots/plot_accel_bias_eval.py \\
      results/tmp/MH_01_easy_accel_bias_eval_1s.csv \\
      --output results/plots/MH_01_easy_accel_bias_eval_1s.png \\
      --max-duration-s 180
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_EXPECTED_COLS = {
    "timestamp_start_s",
    "accel_bias_est_x_mps2",
    "accel_bias_est_y_mps2",
    "accel_bias_est_z_mps2",
    "accel_bias_est_norm_mps2",
    "velocity_error_norm_mps",
}


def _load_csv(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    rows: list[dict] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not _EXPECTED_COLS.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"CSV missing required columns.\n"
                f"Expected subset: {sorted(_EXPECTED_COLS)}\n"
                f"Got: {reader.fieldnames}"
            )
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def _make_plot(
    rows: list[dict],
    max_duration_s: float | None,
    output_path: Path,
) -> None:
    start_s = rows[0]["timestamp_start_s"]

    t:      list[float] = []
    bx:     list[float] = []
    by:     list[float] = []
    bz:     list[float] = []
    bnorm:  list[float] = []
    verr:   list[float] = []

    for row in rows:
        t_rel = row["timestamp_start_s"] - start_s
        if max_duration_s is not None and t_rel > max_duration_s:
            break
        t.append(t_rel)
        bx.append(row["accel_bias_est_x_mps2"])
        by.append(row["accel_bias_est_y_mps2"])
        bz.append(row["accel_bias_est_z_mps2"])
        bnorm.append(row["accel_bias_est_norm_mps2"])
        verr.append(row["velocity_error_norm_mps"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # --- subplot 1: bias components ---
    ax = axes[0]
    ax.plot(t, bx, color="#1f77b4", linewidth=1.0, label="x")
    ax.plot(t, by, color="#ff7f0e", linewidth=1.0, label="y")
    ax.plot(t, bz, color="#2ca02c", linewidth=1.0, label="z")
    ax.axhline(0.0, color="#aaaaaa", linewidth=0.7, linestyle="--")
    ax.set_ylabel("Accel Bias (m/s²)", fontsize=10)
    ax.legend(fontsize=9, loc="upper right", ncol=3)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    # --- subplot 2: bias norm ---
    ax = axes[1]
    ax.plot(t, bnorm, color="#9467bd", linewidth=1.0)
    ax.axhline(0.0, color="#aaaaaa", linewidth=0.7, linestyle="--")
    ax.set_ylabel("Bias Norm (m/s²)", fontsize=10)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    # --- subplot 3: velocity error norm ---
    ax = axes[2]
    ax.plot(t, verr, color="#d62728", linewidth=1.0)
    ax.axhline(0.0, color="#aaaaaa", linewidth=0.7, linestyle="--")
    ax.set_ylabel("Velocity Error (m/s)", fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.suptitle(
        "Offline GT-Based Accelerometer Bias Evaluation",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return t, bx, by, bz, bnorm, verr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot accelerometer bias evaluation results."
    )
    parser.add_argument("csv", help="CSV from evaluate_accel_bias_from_gt")
    parser.add_argument(
        "--output", metavar="PATH",
        help="Output PNG path (default: results/plots/<csv-stem>.png)",
    )
    parser.add_argument(
        "--max-duration-s", type=float, default=None, metavar="FLOAT",
        help="Truncate at this relative time in seconds (default: full)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    try:
        rows = _load_csv(csv_path)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")

    if not rows:
        sys.exit("Error: CSV is empty")

    output_path = Path(args.output) if args.output else \
        Path("results/plots") / f"{csv_path.stem}.png"

    t, bx, by, bz, bnorm, verr = _make_plot(rows, args.max_duration_s, output_path)

    if not t:
        sys.exit("Error: no rows within the requested duration")

    n = len(t)
    mean_bx   = sum(bx)    / n
    mean_by   = sum(by)    / n
    mean_bz   = sum(bz)    / n
    mean_norm = sum(bnorm) / n
    med_norm  = statistics.median(bnorm)
    max_norm  = max(bnorm)
    mean_verr = sum(verr)  / n
    max_verr  = max(verr)

    print(f"  Rows plotted:            {n}")
    print(f"  Duration covered:        {t[-1]:.1f} s")
    print(f"  Mean bias [x, y, z]:     [{mean_bx:.4f}, {mean_by:.4f}, {mean_bz:.4f}] m/s²")
    print(f"  Bias norm [mean/med/max]:{mean_norm:.4f} / {med_norm:.4f} / {max_norm:.4f} m/s²")
    print(f"  Vel error [mean/max]:    {mean_verr:.4f} / {max_verr:.4f} m/s")
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
