"""RL configuration for the Unitree G1 custom tracking task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg


def unitree_g1_custom_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for the custom G1 tracking task."""
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_tracking_custom"
  # Match unitree_rl_mjlab (mjlab uses 30_000).
  cfg.max_iterations = 30001
  return cfg
