"""Unitree G1 custom contact-rich flat tracking environment configuration.

Derives from the custom G1 tracking env (``G1-Tracking-Custom``) as a base for
highly contact-rich motions such as crawling, where many body links (not just
the feet) contact the ground.

The stock G1 collision config makes only the feet frictional (condim=3); every
other body geom is condim=1, i.e. frictionless. This env enables tangential
friction (condim=3) and priority=1 on *all* collision geoms so the whole body
has meaningful, robot-governed ground friction, then randomizes that friction.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.config.g1_custom.env_cfgs import (
  unitree_g1_custom_flat_tracking_env_cfg,
)
from mjlab.utils.spec_config import CollisionCfg

# Whole-body collision: every collision geom gets tangential friction (condim=3)
# and priority=1 (so the robot's friction governs robot-vs-ground contacts, even
# at the low end of the randomized range). Friction itself is set per-episode by
# the ``body_friction`` startup DR term below.
CONTACT_RICH_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim=3,
  priority=1,
)

# Tangential friction range, randomized independently per body geom.
BODY_FRICTION_RANGE = (0.3, 1.6)


def unitree_g1_custom_contactrich_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the contact-rich Unitree G1 (29-DoF) flat tracking config.

  Starts from ``G1-Tracking-Custom`` and (1) makes every body collision geom
  frictional (condim=3, priority=1) and (2) replaces the foot-only friction DR
  with whole-body tangential friction randomization.
  """
  cfg = unitree_g1_custom_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation, play=play
  )

  # Make every body geom frictional and robot-governed (see module docstring).
  cfg.scene.entities["robot"].collisions = (CONTACT_RICH_COLLISION,)

  # Replace the inherited foot-only friction DR with whole-body tangential
  # friction, randomized independently per geom.
  cfg.events.pop("foot_friction", None)
  cfg.events["body_friction"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision",)),
      "operation": "abs",
      "ranges": BODY_FRICTION_RANGE,  # tangential (axis 0) only
      "shared_random": False,
    },
  )

  return cfg
