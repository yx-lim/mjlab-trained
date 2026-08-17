from dataclasses import replace

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  unitree_g1_custom_flat_tracking_env_cfg,
)
from .rl_cfg import (
  unitree_g1_custom_tracking_ppo_runner_cfg,
)

_env_cfg = unitree_g1_custom_flat_tracking_env_cfg(has_state_estimation=False)
_play_env_cfg = unitree_g1_custom_flat_tracking_env_cfg(
  has_state_estimation=False, play=True
)

# Stronger action-rate penalty (-0.1 -> -0.5) to curb high-frequency action chatter on hardware.
# Scoped to THIS task only: ContactRich-Custom and the crawling tasks share the same env builder
# but keep the base -0.1 (they override rewards on their own copies).
for _cfg in (_env_cfg, _play_env_cfg):
  _cfg.rewards["action_rate_l2"] = replace(_cfg.rewards["action_rate_l2"], weight=-0.5)

register_mjlab_task(
  task_id="G1-Tracking-Custom",
  env_cfg=_env_cfg,
  play_env_cfg=_play_env_cfg,
  rl_cfg=unitree_g1_custom_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
