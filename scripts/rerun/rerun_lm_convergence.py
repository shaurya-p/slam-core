"""
Visualize Levenberg-Marquardt convergence on the rigid-alignment demo.

Reads the two CSVs written by demo_lm_point_alignment and animates the
optimizer: the model cloud transformed by the current pose estimate
converging onto the observed cloud, with cost and lambda over iterations.

Timeline: "iteration" (sequence index; row 0 is the initial state).

Usage:
    ./build/tools/demo_lm_point_alignment \\
      results/optim/lm_iterations.csv results/optim/lm_points.csv
    uv run python scripts/rerun/rerun_lm_convergence.py \\
      results/optim/lm_iterations.csv results/optim/lm_points.csv
    rerun results/rerun/lm_point_alignment.rrd

Output: results/rerun/lm_point_alignment.rrd (or --output)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from slam_core_tools.viz import GT_X_COLOR, SERIES_X_COLORS, load_csv_rows
from slam_core_tools.viz.common import mat_from_row

_ITER_COLS = frozenset({
    "iteration", "cost", "lambda", "step_norm", "accepted",
    "r00", "r01", "r02", "r10", "r11", "r12", "r20", "r21", "r22",
    "t_x", "t_y", "t_z",
})
_POINT_COLS = frozenset({"p_x", "p_y", "p_z", "q_x", "q_y", "q_z"})

_OBSERVED_COLOR  = GT_X_COLOR         # blue: observed target cloud
_ESTIMATED_COLOR = SERIES_X_COLORS[0]  # red: model cloud under current estimate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize LM convergence on the rigid-alignment demo."
    )
    parser.add_argument("iterations_csv", help="Iterations CSV from demo_lm_point_alignment")
    parser.add_argument("points_csv",     help="Points CSV from demo_lm_point_alignment")
    parser.add_argument(
        "--output", metavar="PATH",
        help="Output .rrd path (default: results/rerun/lm_point_alignment.rrd)",
    )
    args = parser.parse_args()

    try:
        iters  = load_csv_rows(Path(args.iterations_csv), _ITER_COLS, "Iterations")
        points = load_csv_rows(Path(args.points_csv), _POINT_COLS, "Points")
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")
    if not iters or not points:
        sys.exit("Error: empty input CSV")

    p_B = np.array([[r["p_x"], r["p_y"], r["p_z"]] for r in points])
    q_A = np.array([[r["q_x"], r["q_y"], r["q_z"]] for r in points])

    rrd_path = Path(args.output) if args.output else Path("results/rerun/lm_point_alignment.rrd")
    rrd_path.parent.mkdir(parents=True, exist_ok=True)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="Alignment", origin="alignment"),
            rrb.Vertical(
                rrb.TimeSeriesView(name="Cost (log scale)", origin="cost"),
                rrb.TimeSeriesView(name="Lambda",           origin="lambda"),
            ),
            column_shares=[3, 2],
        ),
    )

    rr.init("slam_core/lm_point_alignment", spawn=False)
    rr.save(str(rrd_path), default_blueprint=blueprint)

    # Observed target cloud is static; the estimated cloud moves per iteration.
    rr.log("alignment/observed", rr.Points3D(q_A, colors=_OBSERVED_COLOR, radii=0.02),
           static=True)

    for row in iters:
        rr.set_time("iteration", sequence=int(row["iteration"]) + 1)  # row -1 -> tick 0
        R = mat_from_row(row)
        t = np.array([row["t_x"], row["t_y"], row["t_z"]])
        transformed = (R @ p_B.T).T + t
        rr.log("alignment/estimated",
               rr.Points3D(transformed, colors=_ESTIMATED_COLOR, radii=0.02))
        # Residual lines from estimate to observation.
        strips = [[transformed[i].tolist(), q_A[i].tolist()] for i in range(len(q_A))]
        rr.log("alignment/residuals",
               rr.LineStrips3D(strips, colors=[[150, 150, 150]], radii=0.002))
        rr.log("cost/half_sq_norm", rr.Scalars(row["cost"]))
        rr.log("lambda/damping",    rr.Scalars(row["lambda"]))

    final = iters[-1]
    print(f"  Iterations logged: {len(iters)} (initial state + LM iterations)")
    print(f"  Cost: {iters[0]['cost']:.6g} -> {final['cost']:.6g}")
    print(f"  Saved: {rrd_path}")


if __name__ == "__main__":
    main()
