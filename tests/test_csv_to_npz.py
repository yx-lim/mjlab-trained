from pathlib import Path

import numpy as np
from scripts.csv_to_npz import DEFAULT_ROBOT_XML, MotionLoader, convert_motion


def _write_test_motion(path: Path) -> None:
  frame_count = 4
  motion = np.zeros((frame_count, 36), dtype=np.float32)
  motion[:, 0] = np.arange(frame_count, dtype=np.float32) * 0.5
  motion[:, 2] = 0.8
  motion[:, 6] = 1.0  # Identity quaternion in xyzw format.
  np.savetxt(path, motion, delimiter=",")


def test_converter_is_standalone_and_preserves_g1_archive_layout(tmp_path: Path):
  csv_path = tmp_path / "motion.csv"
  _write_test_motion(csv_path)

  motion = MotionLoader(csv_path, input_fps=2.0, output_fps=4.0)
  archive, frames = convert_motion(motion, DEFAULT_ROBOT_XML)

  assert frames == []
  assert archive["fps"].tolist() == [4.0]
  assert archive["joint_pos"].shape == (6, 29)
  assert archive["joint_vel"].shape == (6, 29)
  assert archive["body_pos_w"].shape == (6, 30, 3)
  assert archive["body_quat_w"].shape == (6, 30, 4)
  assert archive["body_lin_vel_w"].shape == (6, 30, 3)
  assert archive["body_ang_vel_w"].shape == (6, 30, 3)
  np.testing.assert_allclose(archive["body_pos_w"][:, 0], motion.motion_base_pos)
  np.testing.assert_allclose(archive["body_quat_w"][:, 0, 0], 1.0)
  np.testing.assert_allclose(archive["body_quat_w"][:, 0, 1:], 0.0)
  np.testing.assert_allclose(archive["body_lin_vel_w"][:, 0, 0], 1.0)
