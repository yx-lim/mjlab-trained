#!/usr/bin/env python3
"""Log one mjlab motion-tracking rollout step by step.

Place this file at:

    mjlab/scripts/debug_tracking_step.py

Run it from the mjlab repository root.

Example using mjlab's downloaded demo assets:

    uv run python scripts/debug_tracking_step.py \
        Mjlab-Tracking-Flat-Unitree-G1 \
        --demo \
        --output-file logs/debug_tracking_demo.csv

Example using a local checkpoint and motion:

    uv run python scripts/debug_tracking_step.py \
        Mjlab-Tracking-Flat-Unitree-G1 \
        --checkpoint-file /path/to/model.pt \
        --motion-file /path/to/motion.npz \
        --output-file logs/debug_tracking_step.csv

The CSV contains one row per policy/environment step, including:

- step
- motion_frame
- reference_joint_pos
- actual_joint_pos
- policy_action
- anchor_position_error
- anchor_rotation_error
- body_position_error
- body_rotation_error
- total_reward
- terminated

It also records timeout/done flags, each weighted reward term, and each
termination condition. Vector-valued fields are encoded as compact JSON arrays.
A sidecar ``*.metadata.json`` file records the joint order and run configuration.

Important implementation details:

1. The reference is snapshotted *before* env.step(). mjlab scores the upcoming
   physics result against that current reference frame, then advances the
   motion command after reward computation.

2. ``auto_reset`` is disabled. This preserves the actual terminal robot state
   on a failed step instead of immediately replacing it with a reset state.

3. By default, the script logs the remaining frame-to-frame transitions:
   ``motion_frame_count - start_frame - 1``. It stops before MotionCommand wraps
   to frame zero and teleports the robot to a newly sampled reference state.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.lab_api.math import quat_error_magnitude
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class DebugTrackingConfig:
    """Configuration for a single-environment tracking trace."""

    checkpoint_file: Path | None = None
    """Local trained checkpoint (.pt). Required unless --demo is used."""

    motion_file: Path | None = None
    """Local mjlab motion file (.npz). Required unless --demo is used."""

    output_file: Path = Path("logs/debug_tracking_step.csv")
    """Step-by-step CSV output path."""

    metadata_file: Path | None = None
    """Optional metadata JSON path. Defaults to <output>.metadata.json."""

    demo: bool = False
    """Use mjlab's downloaded pretrained demo checkpoint and motion."""

    start_frame: int = 0
    """Reference frame at which to initialize the robot and policy rollout."""

    max_steps: int | None = None
    """Maximum steps to log. Defaults to all remaining frame transitions."""

    device: str | None = None
    """Execution device. Defaults to CUDA when available, otherwise CPU."""

    disable_terminations: bool = False
    """Disable failure termination terms. Useful for observing full divergence."""

    include_reward_terms: bool = True
    """Add one CSV column for every weighted reward term."""

    include_termination_terms: bool = True
    """Add one CSV column for every individual termination condition."""


def _resolve_assets(cfg: DebugTrackingConfig) -> tuple[Path, Path]:
    """Resolve local or demo checkpoint and motion paths."""
    if cfg.demo:
        if cfg.checkpoint_file is not None or cfg.motion_file is not None:
            raise ValueError(
                "Use either --demo or explicit --checkpoint-file/--motion-file, not both."
            )
        from mjlab.scripts.gcs import ensure_default_checkpoint, ensure_default_motion

        checkpoint = Path(ensure_default_checkpoint())
        motion = Path(ensure_default_motion())
    else:
        if cfg.checkpoint_file is None or cfg.motion_file is None:
            raise ValueError(
                "Provide both --checkpoint-file and --motion-file, or use --demo."
            )
        checkpoint = cfg.checkpoint_file
        motion = cfg.motion_file

    checkpoint = checkpoint.expanduser().resolve()
    motion = motion.expanduser().resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not motion.is_file():
        raise FileNotFoundError(f"Motion file does not exist: {motion}")

    return checkpoint, motion


def _tensor_json(value: torch.Tensor) -> str:
    """Serialize a one-dimensional tensor as a compact JSON array."""
    data = value.detach().to("cpu", dtype=torch.float32).reshape(-1).tolist()
    return json.dumps(data, separators=(",", ":"))


def _scalar(value: torch.Tensor | float | int | bool) -> float:
    """Convert a scalar tensor/value to a Python float."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().to("cpu").item())
    return float(value)


def _safe_names(value: Any, expected_count: int, prefix: str) -> list[str]:
    """Return model-provided names, or stable generated names as a fallback."""
    try:
        names = list(value)
    except (TypeError, AttributeError):
        names = []

    if len(names) != expected_count:
        names = [f"{prefix}_{i:02d}" for i in range(expected_count)]
    return names


def _snapshot_reference(command: MotionCommand) -> SimpleNamespace:
    """Capture the reference that the upcoming physics step is scored against."""
    return SimpleNamespace(
        motion_frame=int(command.time_steps[0].item()),
        joint_pos=command.joint_pos.clone(),
        anchor_pos_w=command.anchor_pos_w.clone(),
        anchor_quat_w=command.anchor_quat_w.clone(),
        body_pos_relative_w=command.body_pos_relative_w.clone(),
        body_quat_relative_w=command.body_quat_relative_w.clone(),
    )


def _compute_tracking_errors(
    reference: SimpleNamespace,
    command: MotionCommand,
) -> dict[str, float]:
    """Compare the snapshotted reference with the post-step robot state."""
    anchor_position_error = torch.linalg.vector_norm(
        reference.anchor_pos_w - command.robot_anchor_pos_w,
        dim=-1,
    )

    anchor_rotation_error = quat_error_magnitude(
        reference.anchor_quat_w,
        command.robot_anchor_quat_w,
    )

    body_position_error = torch.linalg.vector_norm(
        reference.body_pos_relative_w - command.robot_body_pos_w,
        dim=-1,
    ).mean(dim=-1)

    body_rotation_error = quat_error_magnitude(
        reference.body_quat_relative_w,
        command.robot_body_quat_w,
    ).mean(dim=-1)

    return {
        "anchor_position_error": _scalar(anchor_position_error[0]),
        "anchor_rotation_error": _scalar(anchor_rotation_error[0]),
        "body_position_error": _scalar(body_position_error[0]),
        "body_rotation_error": _scalar(body_rotation_error[0]),
    }


def _manager_term_values(manager: Any, env_idx: int = 0) -> dict[str, float]:
    """Read public per-term values exposed by reward/termination managers."""
    result: dict[str, float] = {}
    for name, values in manager.get_active_iterable_terms(env_idx):
        if not values:
            continue
        result[name] = float(values[0])
    return result


def _metadata_path(cfg: DebugTrackingConfig) -> Path:
    if cfg.metadata_file is not None:
        return cfg.metadata_file.expanduser().resolve()
    output = cfg.output_file.expanduser().resolve()
    return output.with_suffix(output.suffix + ".metadata.json")


def run_debug_tracking(task_id: str, cfg: DebugTrackingConfig) -> Path:
    """Run one deterministic rollout and write a detailed CSV trace."""
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_path, motion_path = _resolve_assets(cfg)

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    motion_cfg = env_cfg.commands.get("motion")
    if not isinstance(motion_cfg, MotionCommandCfg):
        raise ValueError(f"Task {task_id!r} is not a motion-tracking task.")

    # Deterministic, single-environment debugging.
    motion_cfg.motion_file = str(motion_path)
    motion_cfg.sampling_mode = "start"
    motion_cfg.pose_range = {}
    motion_cfg.velocity_range = {}
    env_cfg.scene.num_envs = 1
    env_cfg.auto_reset = False

    if cfg.disable_terminations:
        env_cfg.terminations = {}

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint_path),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    command = cast(
        MotionCommand,
        env.unwrapped.command_manager.get_term("motion"),
    )

    frame_count = int(command.motion.time_step_total)
    if not 0 <= cfg.start_frame < frame_count:
        raise ValueError(
            f"--start-frame must be in [0, {frame_count - 1}], "
            f"received {cfg.start_frame}."
        )

    available_steps = frame_count - cfg.start_frame - 1
    if available_steps <= 0:
        raise ValueError(
            "The selected start frame has no following frame transition to trace."
        )

    if cfg.max_steps is None:
        requested_steps = available_steps
    else:
        if cfg.max_steps <= 0:
            raise ValueError("--max-steps must be positive.")
        requested_steps = min(cfg.max_steps, available_steps)

    # Reset exactly to the selected reference state, with no RSI perturbation.
    env_ids = torch.tensor([0], dtype=torch.int64, device=env.unwrapped.device)
    env.unwrapped.reset(env_ids=env_ids)
    command.reset_to_frame(env_ids, cfg.start_frame)
    command.update_relative_body_poses()
    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.forward()
    env.unwrapped.sim.sense()
    obs = env.get_observations()

    joint_count = int(command.joint_pos.shape[-1])
    joint_names = _safe_names(
        getattr(command.robot, "joint_names", None),
        joint_count,
        "joint",
    )

    reward_names = (
        list(env.unwrapped.reward_manager.active_terms)
        if cfg.include_reward_terms
        else []
    )
    termination_names = (
        list(env.unwrapped.termination_manager.active_terms)
        if cfg.include_termination_terms
        else []
    )

    base_fields = [
        "step",
        "sim_time_s",
        "motion_frame",
        "reference_joint_pos",
        "actual_joint_pos",
        "policy_action",
        "anchor_position_error",
        "anchor_rotation_error",
        "body_position_error",
        "body_rotation_error",
        "total_reward",
        "total_reward_rate",
        "terminated",
        "truncated",
        "done",
    ]
    reward_fields = [f"reward/{name}" for name in reward_names]
    termination_fields = [f"termination/{name}" for name in termination_names]
    fieldnames = base_fields + reward_fields + termination_fields

    output_path = cfg.output_file.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    final_status = "max_steps"
    final_frame: int | None = None

    try:
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for step in range(requested_steps):
                # Reward is evaluated against this frame. MotionCommand advances
                # to the next frame near the end of env.step().
                reference = _snapshot_reference(command)

                with torch.inference_mode():
                    action = policy(obs)

                obs, reward, dones, _extras = env.step(action)

                errors = _compute_tracking_errors(reference, command)
                terminated = bool(
                    env.unwrapped.termination_manager.terminated[0].item()
                )
                truncated = bool(
                    env.unwrapped.termination_manager.time_outs[0].item()
                )
                done = bool(dones[0].item())

                reward_terms = (
                    _manager_term_values(env.unwrapped.reward_manager)
                    if cfg.include_reward_terms
                    else {}
                )
                termination_terms = (
                    _manager_term_values(env.unwrapped.termination_manager)
                    if cfg.include_termination_terms
                    else {}
                )

                row: dict[str, Any] = {
                    "step": step,
                    "sim_time_s": step * env.unwrapped.step_dt,
                    "motion_frame": reference.motion_frame,
                    "reference_joint_pos": _tensor_json(reference.joint_pos[0]),
                    "actual_joint_pos": _tensor_json(command.robot_joint_pos[0]),
                    "policy_action": _tensor_json(action[0]),
                    **errors,
                    # The environment return is the dt-scaled aggregate reward.
                    "total_reward": _scalar(reward[0]),
                    # Reward manager term values are weighted rates, before dt.
                    "total_reward_rate": _scalar(reward[0]) / env.unwrapped.step_dt,
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "done": int(done),
                }

                for name in reward_names:
                    row[f"reward/{name}"] = reward_terms.get(name, 0.0)
                for name in termination_names:
                    row[f"termination/{name}"] = termination_terms.get(name, 0.0)

                writer.writerow(row)
                rows_written += 1
                final_frame = reference.motion_frame

                if done:
                    final_status = "terminated" if terminated else "truncated"
                    break
            else:
                if requested_steps == available_steps:
                    final_status = "motion_complete"
    finally:
        env.close()

    metadata_path = _metadata_path(cfg)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "task_id": task_id,
        "checkpoint_file": str(checkpoint_path),
        "motion_file": str(motion_path),
        "output_file": str(output_path),
        "device": device,
        "start_frame": cfg.start_frame,
        "motion_frame_count": frame_count,
        "requested_steps": requested_steps,
        "rows_written": rows_written,
        "final_status": final_status,
        "final_logged_motion_frame": final_frame,
        "physics_dt_s": env_cfg.sim.mujoco.timestep,
        "decimation": env_cfg.decimation,
        "environment_step_dt_s": env_cfg.sim.mujoco.timestep * env_cfg.decimation,
        "control_frequency_hz": 1.0
        / (env_cfg.sim.mujoco.timestep * env_cfg.decimation),
        "anchor_body_name": command.cfg.anchor_body_name,
        "tracked_body_names": list(command.cfg.body_names),
        "joint_names": joint_names,
        "array_column_format": "compact JSON array in joint_names order",
        "reward_columns": reward_fields,
        "termination_columns": termination_fields,
        "notes": {
            "total_reward": "dt-scaled aggregate returned by env.step()",
            "total_reward_rate": "total_reward divided by environment step dt",
            "reward_terms": "weighted per-step reward rates before dt scaling",
            "errors": "reference snapshotted before env.step, robot state read after physics",
            "auto_reset": False,
        },
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\n" + "=" * 72)
    print("Tracking trace complete")
    print("=" * 72)
    print(f"CSV:       {output_path}")
    print(f"Metadata:  {metadata_path}")
    print(f"Rows:      {rows_written}")
    print(f"Status:    {final_status}")
    if final_frame is not None:
        print(f"Last frame:{final_frame}")
    print("=" * 72)

    return output_path


def main() -> None:
    # Populate the mjlab task registry.
    import mjlab.tasks  # noqa: F401

    tracking_tasks = [task for task in list_tasks() if "Tracking" in task]
    if not tracking_tasks:
        raise RuntimeError("No tracking tasks are registered.")

    task_id, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(tracking_tasks),
        add_help=False,
        return_unknown_args=True,
        config=__import__("mjlab").TYRO_FLAGS,
    )

    cfg = tyro.cli(
        DebugTrackingConfig,
        args=remaining_args,
        prog=f"{sys.argv[0]} {task_id}",
        config=__import__("mjlab").TYRO_FLAGS,
    )
    run_debug_tracking(task_id, cfg)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
