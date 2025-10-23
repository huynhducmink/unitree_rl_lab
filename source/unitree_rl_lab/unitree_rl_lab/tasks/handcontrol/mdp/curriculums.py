from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)

# --- PATCH: curriculums.py (append) ---

# curriculums.py
import torch

def _success_rate(env, env_ids, reward_term_name="touch_ball", threshold=0.5) -> float:
    """Return mean(success) over env_ids where success = (episode_sum > threshold)."""
    rm = getattr(env, "reward_manager", None)
    sums = getattr(rm, "_episode_sums", None)
    if sums is None or reward_term_name not in sums:
        return 0.0
    # normalize env_ids to a LongTensor on the right device
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    vals = sums[reward_term_name].index_select(0, ids)
    return ((vals > threshold).float().mean()).item()

def reach_cmd_levels(env, env_ids, reward_term_name: str = "touch_ball") -> torch.Tensor:
    """Expand ball spawn corridor as success improves."""
    sr = _success_rate(env, env_ids, reward_term_name, threshold=0.5)
    # only adjust at episode boundaries to avoid thrashing
    if (env.common_step_counter % env.max_episode_length) == 0 and sr > 0.6:
        # get the *config* used by the respawn event and widen ranges a bit
        ev_cfg = env.event_manager.get_term_cfg("respawn_ball")  # same call you used
        xr0, xr1 = ev_cfg.params["x_range"]
        yr0, yr1 = ev_cfg.params["y_range"]
        ev_cfg.params["x_range"] = (xr0, min(xr1 + 0.05, 1.30))
        ev_cfg.params["y_range"] = (max(yr0 - 0.02, -0.35), min(yr1 + 0.02, 0.35))
    return torch.tensor(1.0, device=env.device)

def table_height_levels(env, env_ids, reward_term_name: str = "touch_ball") -> torch.Tensor:
    """Lower table a bit (harder) as the policy gets better; keep ball above surface."""
    sr = _success_rate(env, env_ids, reward_term_name, threshold=0.5)
    if (env.common_step_counter % env.max_episode_length) == 0 and sr > 0.7:
        tbl_cfg = env.scene.cfg.table.spawn
        # table Cube height = size.z; table origin at its center; current z is H/2
        new_center_z = max(tbl_cfg.position[2] - 0.01, 0.30)  # don’t go too low
        tbl_cfg.position = (tbl_cfg.position[0], tbl_cfg.position[1], new_center_z)

        # keep ball slightly above tabletop: top_z = 2*center_z = table height
        ball_cfg = env.scene.cfg.ball.spawn
        ball_cfg.position = (ball_cfg.position[0], ball_cfg.position[1], new_center_z * 2 + 0.07)
    return torch.tensor(1.0, device=env.device)
