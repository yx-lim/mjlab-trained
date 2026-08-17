"""RL configuration for the Unitree G1 custom contact-rich tracking task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.tracking.config.g1_custom.rl_cfg import (
  unitree_g1_custom_tracking_ppo_runner_cfg,
)


def unitree_g1_custom_contactrich_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for the contact-rich G1 tracking task."""
  cfg = unitree_g1_custom_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_tracking_contactrich_custom"
  return cfg
