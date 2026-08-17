"""Shared domain-randomization base for the custom Unitree G1 environments.

A single, tunable set of startup (physical-property) randomization terms that
both the custom tracking and custom velocity envs layer on top of their
inherited base config. Tune magnitudes per-env by passing a ``CustomDRCfg``;
leave perturbation (``push_robot``) and reset randomization to the base config,
which already matches unitree_rl_mjlab.

The terms applied here are all ``startup`` mode, matching mjlab's convention of
keeping physical-property DR active in play mode (only ``push_robot`` and
observation noise are disabled there).
"""

import math
from dataclasses import dataclass, field, replace

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg


@dataclass
class CustomDRCfg:
  """Tunable ranges for the shared custom G1 domain randomization.

  Defaults match the custom *tracking* spec (unitree_rl_mjlab-aligned). The
  velocity env can override the fields that differ (e.g. ``foot_friction``,
  ``encoder_bias``) when it adopts this base.
  """

  # Base-COM offset (m), per axis {0: x, 1: y, 2: z}. Added to the nominal COM.
  base_com: dict[int, tuple[float, float]] = field(
    default_factory=lambda: {
      0: (-0.05, 0.05),
      1: (-0.05, 0.05),
      2: (-0.05, 0.05),
    }
  )
  # PD gain scale factors (multiplicative).
  pd_kp: tuple[float, float] = (0.8, 1.2)
  pd_kd: tuple[float, float] = (0.8, 1.2)
  # Joint friction loss, added to nominal (N·m).
  joint_friction: tuple[float, float] = (0.0, 1.0)
  # Joint armature scale factor (multiplicative).
  joint_armature: tuple[float, float] = (0.95, 1.05)
  # Base (torso_link) mass scale factor (multiplicative). Applied via
  # pseudo_inertia so mass and inertia scale together (a density change).
  base_mass: tuple[float, float] = (0.9, 1.1)
  # Per-joint encoder offset (rad).
  encoder_bias: tuple[float, float] = (-0.015, 0.015)
  # Foot friction coefficient (absolute), shared across foot geoms.
  foot_friction: tuple[float, float] = (0.3, 1.6)


def add_custom_g1_dr(
  cfg: ManagerBasedRlEnvCfg,
  dr_cfg: CustomDRCfg | None = None,
) -> None:
  """Apply the shared custom G1 DR base in place.

  Retunes the inherited ``base_com``, ``encoder_bias``, and ``foot_friction``
  startup terms, and adds ``pd_gains``, ``joint_friction``, ``joint_armature``,
  and ``base_mass`` (torso mass+inertia). Perturbation (``push_robot``) and reset
  randomization are intentionally left to the inherited base config.

  Args:
    cfg: An already-built env config. Its ``events`` dict must already contain
      the ``base_com``, ``encoder_bias``, and ``foot_friction`` terms from the
      base config (both the tracking and velocity bases provide these).
    dr_cfg: Tunable ranges. Defaults to :class:`CustomDRCfg`.
  """
  if dr_cfg is None:
    dr_cfg = CustomDRCfg()

  # Retune inherited startup terms (preserves the base's per-robot targeting,
  # e.g. foot geom names and the base-COM body name).
  cfg.events["base_com"].params["ranges"] = dict(dr_cfg.base_com)
  cfg.events["encoder_bias"].params["bias_range"] = dr_cfg.encoder_bias
  cfg.events["foot_friction"].params["ranges"] = dr_cfg.foot_friction

  # Add new startup terms (joint/actuator regex is robot-agnostic).
  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_names=[".*"]),
      "kp_range": dr_cfg.pd_kp,
      "kd_range": dr_cfg.pd_kd,
      "operation": "scale",
    },
  )
  cfg.events["joint_friction"] = EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "ranges": dr_cfg.joint_friction,
      "operation": "add",
    },
  )
  cfg.events["joint_armature"] = EventTermCfg(
    mode="startup",
    func=dr.joint_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "ranges": dr_cfg.joint_armature,
      "operation": "scale",
    },
  )
  # Randomize torso mass and inertia together (a density change). pseudo_inertia
  # scales mass by e^(2*alpha), so a multiplicative mass-scale range (lo, hi)
  # maps to alpha_range = (0.5*ln(lo), 0.5*ln(hi)).
  mass_lo, mass_hi = dr_cfg.base_mass
  cfg.events["base_mass"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      "alpha_range": (0.5 * math.log(mass_lo), 0.5 * math.log(mass_hi)),
    },
  )


def add_custom_g1_actuator_delay(
  cfg: ManagerBasedRlEnvCfg,
  min_delay_sec: float = 0.0,
  max_delay_sec: float = 0.02,
) -> None:
  """Add per-env command delay to the G1 built-in position actuators in place.

  Actuator delay is a built-in actuator property, not an event term: each step a
  per-env lag is sampled uniformly in ``[min, max]`` physics steps and applied to
  the command targets, modeling policy-to-motor latency. Delays are given in
  seconds here and converted to physics steps via the sim timestep, and the lag is
  re-sampled once per control step.

  Rebuilds the robot entity functionally (via ``dataclasses.replace``) so the
  shared G1 actuator constants in ``g1_constants`` are left untouched.

  Args:
    cfg: An already-built env config whose ``robot`` entity uses the G1 built-in
      position actuators.
    min_delay_sec: Minimum command delay in seconds.
    max_delay_sec: Maximum command delay in seconds. ``0`` disables delay.
  """
  robot_cfg = cfg.scene.entities["robot"]
  assert isinstance(robot_cfg, EntityCfg)
  assert robot_cfg.articulation is not None

  timestep = cfg.sim.mujoco.timestep
  min_lag = round(min_delay_sec / timestep)
  max_lag = round(max_delay_sec / timestep)

  actuators = tuple(
    replace(
      act,
      delay_min_lag=min_lag,
      delay_max_lag=max_lag,
      delay_update_period=cfg.decimation,
      delay_per_env_phase=True,
    )
    if isinstance(act, BuiltinPositionActuatorCfg)
    else act
    for act in robot_cfg.articulation.actuators
  )
  cfg.scene.entities["robot"] = replace(
    robot_cfg,
    articulation=replace(robot_cfg.articulation, actuators=actuators),
  )
