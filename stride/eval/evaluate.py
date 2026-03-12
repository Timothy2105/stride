"""Evaluate a trained MLP BC policy in Adroit environments.

Capabilities
------------
- Roll out policy for N episodes with deterministic seeds
- Render every frame and save rollout videos as MP4
- Compute per-episode reward, success, and episode length
- Aggregate statistics (mean, std, success rate)
- Log everything to wandb (metrics + videos)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — registers Adroit envs
import numpy as np

logger = logging.getLogger(__name__)


def _save_video(frames: list[np.ndarray], path: str, fps: int = 30) -> None:
    """Save a list of RGB frames as an MP4 video."""
    import imageio

    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                output_params=["-crf", "23"])
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def evaluate_policy(
    policy,
    env_name: str,
    n_episodes: int = 20,
    seed: int = 42,
    render: bool = True,
    video_dir: str | None = None,
    max_episode_steps: int = 400,
    wandb_run=None,
    log_prefix: str = "eval",
    verbose: bool = True,
) -> dict:
    """Roll out a policy in a gymnasium environment and collect metrics.

    Parameters
    ----------
    policy          : MLPPolicy (or any object with ``get_action(obs_np) → act_np``).
    env_name        : gymnasium environment id (e.g. ``AdroitHandPen-v1``).
    n_episodes      : number of evaluation episodes.
    seed            : base seed (episode i uses seed + i).
    render          : whether to capture RGB frames for video.
    video_dir       : directory to save episode MP4s (required if render=True).
    max_episode_steps : hard cutoff per episode.
    wandb_run       : active wandb run for logging (or None).
    log_prefix      : prefix for logged metric keys.
    verbose         : print per-episode summaries.

    Returns
    -------
    dict with:
        per_episode : list[dict] — per-episode reward, success, length, video_path
        mean_reward, std_reward, success_rate, mean_length : aggregate stats
    """
    import torch

    device = "cpu"
    if hasattr(policy, "_obs_mean"):
        device = str(policy._obs_mean.device)

    render_mode = "rgb_array" if render else None
    env = gym.make(env_name, render_mode=render_mode, max_episode_steps=max_episode_steps)

    per_episode: list[dict] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        # Handle dict observations (gymnasium-robotics may return dicts)
        if isinstance(obs, dict):
            obs = obs.get("observation", obs.get("state", np.concatenate(list(obs.values()))))

        frames: list[np.ndarray] = []
        episode_reward = 0.0
        success = False
        step = 0

        done = False
        while not done:
            action = policy.get_action(obs)
            obs_next, reward, terminated, truncated, info = env.step(action)

            if isinstance(obs_next, dict):
                obs_next = obs_next.get("observation", obs_next.get(
                    "state", np.concatenate(list(obs_next.values()))))

            episode_reward += float(reward)
            # Adroit environments report success in info
            if info.get("success", False) or info.get("is_success", False):
                success = True

            if render:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            obs = obs_next
            step += 1
            done = terminated or truncated

        ep_info: dict = {
            "episode": ep,
            "reward": episode_reward,
            "success": float(success),
            "length": step,
        }

        # Save video
        if render and frames and video_dir is not None:
            vpath = os.path.join(video_dir, f"ep_{ep:03d}.mp4")
            _save_video(frames, vpath)
            ep_info["video_path"] = vpath

        per_episode.append(ep_info)

        if verbose:
            logger.info(
                f"[{log_prefix}] ep {ep:3d}  reward={episode_reward:8.2f}  "
                f"success={success}  len={step}"
            )

    env.close()

    # ---- aggregate stats --------------------------------------------------
    rewards = np.array([e["reward"] for e in per_episode])
    successes = np.array([e["success"] for e in per_episode])
    lengths = np.array([e["length"] for e in per_episode])

    stats = {
        "per_episode": per_episode,
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "success_rate": float(successes.mean()),
        "mean_length": float(lengths.mean()),
        "n_episodes": n_episodes,
    }

    if verbose:
        logger.info(
            f"[{log_prefix}] {n_episodes} episodes  "
            f"reward={stats['mean_reward']:.2f}±{stats['std_reward']:.2f}  "
            f"success={stats['success_rate']:.1%}  "
            f"length={stats['mean_length']:.0f}"
        )

    # ---- wandb logging ----------------------------------------------------
    if wandb_run is not None:
        wandb_run.log({
            f"{log_prefix}/mean_reward": stats["mean_reward"],
            f"{log_prefix}/std_reward": stats["std_reward"],
            f"{log_prefix}/success_rate": stats["success_rate"],
            f"{log_prefix}/mean_length": stats["mean_length"],
        })

        # Log individual episode metrics as a wandb table
        try:
            import wandb

            table = wandb.Table(columns=["episode", "reward", "success", "length"])
            for e in per_episode:
                table.add_data(e["episode"], e["reward"], e["success"], e["length"])
            wandb_run.log({f"{log_prefix}/episodes": table})

            # Log videos
            if render and video_dir is not None:
                for e in per_episode:
                    vp = e.get("video_path")
                    if vp and os.path.exists(vp):
                        wandb_run.log({
                            f"{log_prefix}/video_ep{e['episode']:03d}":
                                wandb.Video(vp, format="mp4"),
                        })
        except Exception as exc:
            logger.warning(f"[{log_prefix}] wandb video/table logging failed: {exc}")

    return stats


def rollout_for_scoring(
    policy,
    env_name: str,
    n_episodes: int = 100,
    seed: int = 0,
    max_episode_steps: int = 400,
    verbose: bool = True,
) -> dict:
    """Roll out a policy and collect transition data for TRAK scoring.

    Returns a dict in the same format as the training data:
        observations, actions, rewards, episode_ends, successes
    """
    env = gym.make(env_name, max_episode_steps=max_episode_steps)

    all_obs: list[np.ndarray] = []
    all_act: list[np.ndarray] = []
    all_rew: list[np.ndarray] = []
    episode_ends: list[int] = []
    successes: list[bool] = []
    cursor = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        if isinstance(obs, dict):
            obs = obs.get("observation", obs.get("state", np.concatenate(list(obs.values()))))

        ep_obs, ep_act, ep_rew = [], [], []
        success = False
        done = False

        while not done:
            action = policy.get_action(obs)
            obs_next, reward, terminated, truncated, info = env.step(action)
            if isinstance(obs_next, dict):
                obs_next = obs_next.get("observation", obs_next.get(
                    "state", np.concatenate(list(obs_next.values()))))

            ep_obs.append(obs.astype(np.float32))
            ep_act.append(action.astype(np.float32))
            ep_rew.append(float(reward))

            if info.get("success", False) or info.get("is_success", False):
                success = True

            obs = obs_next
            done = terminated or truncated

        T = len(ep_obs)
        all_obs.append(np.stack(ep_obs))
        all_act.append(np.stack(ep_act))
        all_rew.append(np.array(ep_rew, dtype=np.float32))
        cursor += T
        episode_ends.append(cursor)
        successes.append(success)

        if verbose and (ep % 20 == 0 or ep == n_episodes - 1):
            logger.info(
                f"[rollout] ep {ep:3d}/{n_episodes}  len={T}  "
                f"success={success}  reward={sum(ep_rew):.1f}"
            )

    env.close()

    return {
        "observations": np.concatenate(all_obs, axis=0),
        "actions": np.concatenate(all_act, axis=0),
        "rewards": np.concatenate(all_rew, axis=0),
        "episode_ends": episode_ends,
        "successes": np.array(successes, dtype=bool),
    }
