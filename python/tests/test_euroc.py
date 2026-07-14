"""Tests for slam_core_tools.datasets.euroc CSV loaders."""

import math

import pytest

from slam_core_tools.datasets.euroc import (
    associate_stereo_pairs,
    CameraFrame,
    GroundTruthSample,
    nearest_gt_sample,
    read_cam_csv,
    read_groundtruth_csv,
    read_imu_csv,
    validate_imu,
)

IMU_HEADER = (
    "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],"
    "w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],"
    "a_RS_S_z [m s^-2]\n"
)

GT_HEADER = (
    "#timestamp,p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],"
    "q_RS_x [],q_RS_y [],q_RS_z [],v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],"
    "v_RS_R_z [m s^-1]\n"
)


def test_read_imu_csv_parses_units_and_order(tmp_path):
    path = tmp_path / "imu.csv"
    path.write_text(
        IMU_HEADER
        + "1403636579758555392,-0.1,0.2,0.3,8.0,-0.5,-2.0\n"
        + "1403636579763555584,0.4,-0.5,0.6,8.1,-0.6,-2.1\n"
    )
    samples = read_imu_csv(path)
    assert len(samples) == 2
    assert samples[0].timestamp_s == pytest.approx(1403636579758555392 * 1e-9)
    assert samples[0].gyro_radps == (-0.1, 0.2, 0.3)
    assert samples[0].accel_mps2 == (8.0, -0.5, -2.0)
    assert samples[0].timestamp_s < samples[1].timestamp_s


def test_read_imu_csv_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_imu_csv(tmp_path / "missing.csv")


def test_validate_imu_flags_nonfinite_and_nonmonotonic(tmp_path):
    path = tmp_path / "imu.csv"
    path.write_text(
        IMU_HEADER
        + "2000000000,0.1,0.2,0.3,1.0,2.0,3.0\n"
        + "1000000000,0.1,0.2,nan,1.0,2.0,3.0\n"
    )
    warnings = validate_imu(read_imu_csv(path))
    assert len(warnings) == 2
    assert any("non-finite" in w for w in warnings)
    assert any("non-monotonic" in w for w in warnings)


def test_read_groundtruth_csv_parses_pose_and_velocity(tmp_path):
    path = tmp_path / "gt.csv"
    path.write_text(
        GT_HEADER
        + "1403636580838555648,4.6,-1.8,0.8,1.0,0.0,0.0,0.0,0.01,-0.02,0.03\n"
        + "1403636580843555648,4.7,-1.9,0.9,0.0,1.0\n"  # short row skipped
    )
    samples = read_groundtruth_csv(path)
    assert len(samples) == 1
    s = samples[0]
    assert s.timestamp_s == pytest.approx(1403636580838555648 * 1e-9)
    assert (s.p_x, s.p_y, s.p_z) == (4.6, -1.8, 0.8)
    assert (s.q_w, s.q_x, s.q_y, s.q_z) == (1.0, 0.0, 0.0, 0.0)
    assert (s.v_x, s.v_y, s.v_z) == (0.01, -0.02, 0.03)


def _gt_at(ts: float) -> GroundTruthSample:
    return GroundTruthSample(
        timestamp_s=ts, p_x=0, p_y=0, p_z=0,
        q_w=1, q_x=0, q_y=0, q_z=0, v_x=0, v_y=0, v_z=0,
    )


def test_nearest_gt_sample_clamps_and_ties_earlier():
    samples = [_gt_at(1.0), _gt_at(2.0), _gt_at(4.0)]
    assert nearest_gt_sample(0.0, samples).timestamp_s == 1.0
    assert nearest_gt_sample(9.0, samples).timestamp_s == 4.0
    assert nearest_gt_sample(1.4, samples).timestamp_s == 1.0
    assert nearest_gt_sample(1.6, samples).timestamp_s == 2.0
    assert nearest_gt_sample(3.0, samples).timestamp_s == 2.0  # tie -> earlier
    assert nearest_gt_sample(1.0, []) is None


def test_read_cam_csv_and_stereo_association(tmp_path):
    cam0 = tmp_path / "cam0.csv"
    cam1 = tmp_path / "cam1.csv"
    cam0.write_text(
        "#timestamp [ns],filename\n"
        "1000000000,a.png\n"
        "2000000000,b.png\n"
    )
    cam1.write_text(
        "#timestamp [ns],filename\n"
        "1000000000,c.png\n"
        "3000000000,d.png\n"
    )
    frames0 = read_cam_csv(cam0)
    frames1 = read_cam_csv(cam1)
    assert frames0[0] == CameraFrame(timestamp_s=1.0, filename="a.png")

    pairs = associate_stereo_pairs(frames0, frames1)
    assert len(pairs) == 1
    assert pairs[0].timestamp_s == 1.0
    assert pairs[0].filename_cam0 == "a.png"
    assert pairs[0].filename_cam1 == "c.png"
