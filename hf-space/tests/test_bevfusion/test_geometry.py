"""BEV projection geometry: known extrinsic -> known pixel / known BEV cell."""

import math

import numpy as np

from sensorflow.bevfusion.geometry import (
    BEVGrid, backproject, make_camera_extrinsic, make_intrinsics, project_to_image,
)


def test_known_extrinsic_projects_optical_axis_to_principal_point():
    K = make_intrinsics()  # fx=fy=500, cx=400, cy=225
    T = make_camera_extrinsic(tx=0.2, ty=0.0, tz=1.5)
    # A point straight down the optical axis maps exactly to the principal point.
    u, v, rng = project_to_image(np.array([20.2, 0.0, 1.5]), T, K)
    assert abs(u - 400.0) < 1e-9
    assert abs(v - 225.0) < 1e-9
    assert abs(rng - 20.0) < 1e-9


def test_known_extrinsic_off_axis_pixel_matches_analytic_formula():
    K = make_intrinsics()
    T = make_camera_extrinsic(tx=0.2, ty=0.0, tz=1.5)
    px, py, pz = 10.2, 2.0, 0.8
    u, v, rng = project_to_image(np.array([px, py, pz]), T, K)
    # For this extrinsic: x_cam = -(y-ty), y_cam = -(z-tz), z_cam = x-tx.
    x_c, y_c, z_c = -(py - 0.0), -(pz - 1.5), px - 0.2
    assert abs(u - (500.0 * x_c / z_c + 400.0)) < 1e-9
    assert abs(v - (500.0 * y_c / z_c + 225.0)) < 1e-9
    assert abs(rng - math.sqrt(x_c**2 + y_c**2 + z_c**2)) < 1e-9


def test_backprojection_round_trip_recovers_ego_point():
    K = make_intrinsics()
    T = make_camera_extrinsic(tx=0.2, ty=0.0, tz=1.5, yaw=0.05)
    point = np.array([34.0, -6.5, 1.1])
    u, v, rng = project_to_image(point, T, K)
    recovered = backproject(u, v, rng, T, K)
    np.testing.assert_allclose(recovered, point, atol=1e-9)


def test_point_behind_camera_is_rejected():
    K = make_intrinsics()
    T = make_camera_extrinsic()
    assert project_to_image(np.array([-5.0, 0.0, 1.5]), T, K) is None


def test_known_bev_cell_and_center():
    grid = BEVGrid(x_min=0.0, x_max=80.0, y_min=-25.0, y_max=25.0, cell=0.5)
    assert grid.nx == 160 and grid.ny == 100
    # (30.3, 2.1): ix = 30.3/0.5 = 60, iy = (2.1+25)/0.5 = 54
    assert grid.index(30.3, 2.1) == (60, 54)
    cx, cy = grid.center(60, 54)
    assert abs(cx - 30.25) < 1e-9 and abs(cy - 2.25) < 1e-9
    assert grid.index(90.0, 0.0) is None
    assert grid.index(10.0, -30.0) is None
