"""Policy evaluation via rollout in the AdroitHandPen-v1 gymnasium environment.

Runs N episodes with a trained deterministic BC policy and reports:
  - Mean ± std cumulative reward
  - Success rate (fraction of episodes where the task is solved)

AdroitHandPen-v1 reports a 'success' info key in its step() return;
we check for it as a proxy for task completion.
"""

from __future__ import annotations

import os
import numpy as np
import torch


def load_policy(ckpt_path: str):
    """Load a BCPolicy from a checkpoint file."""
    from stride.models.policy import BCPolicy
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hidden = tuple(ckpt.get("hidden", [256, 256]))
    policy = BCPolicy(obs_dim=ckpt["obs_dim"], act_dim=ckpt["act_dim"], hidden=hidden)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    # Stats metadata
    policy.obs_mean = ckpt.get("obs_mean", None)
    policy.obs_std = ckpt.get("obs_std", None)
    policy.latent_mean = ckpt.get("latent_mean", None)
    policy.latent_std = ckpt.get("latent_std", None)
    
    # LSPO flags
    policy.use_lspo = ckpt.get("use_lspo", False)
    policy.use_lspo_norm = ckpt.get("use_lspo_norm", True)
    policy.vae_ckpt = ckpt.get("vae_ckpt", None)
    return policy


@torch.no_grad()
def evaluate_policy(
    policy,
    n_episodes: int = 50,
    env_name: str = "AdroitHandPen-v1",
    seed: int = 0,
    render: bool = False,
    device_str: str = "cpu",
) -> dict[str, float]:
    """Roll out a policy for n_episodes and return performance statistics.

    Parameters
    ----------
    policy     : BCPolicy (or any callable obs→action Tensor)
    n_episodes : number of evaluation episodes
    env_name   : gymnasium environment ID
    seed       : base random seed (each episode gets seed + episode_idx)
    render     : whether to render the environment
    device_str : torch device for policy inference

    Returns
    -------
    dict with keys:
        'mean_reward'  : float
        'std_reward'   : float
        'min_reward'   : float
        'max_reward'   : float
        'success_rate' : float  (fraction of successful episodes)
        'rewards'      : list of per-episode total rewards
    """
    import gymnasium as gym
    try:
        import gymnasium_robotics  # registers AdroitHandPen-v1 and other Adroit envs
    except ImportError:
        pass

    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)

    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu"
                          else "cpu")
    policy = policy.to(device)

    episode_rewards: list[float] = []
    successes: list[bool] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False
        episode_success = False

        while not done:
            obs_raw = np.array(obs, dtype=np.float32)
            
            # Apply observation normalization if stats are available
            obs_mean = getattr(policy, "obs_mean", None)
            obs_std = getattr(policy, "obs_std", None)
            if obs_mean is not None and obs_std is not None:
                obs_raw = (obs_raw - obs_mean.numpy()) / (obs_std.numpy() + 1e-6)

            obs_t = torch.from_numpy(obs_raw).unsqueeze(0).to(device)
            pred = policy(obs_t)

            # LSPO: If policy outputs latent z, decode it to action a
            if getattr(policy, "use_lspo", False):
                if not hasattr(policy, "_vae"):
                    from stride.models.vae import ConditionalVAE
                    vae_ckpt = policy.vae_ckpt
                    if vae_ckpt is None:
                        raise ValueError("Policy marked as LSPO but no vae_ckpt found.")
                    ckpt = torch.load(vae_ckpt, map_location="cpu")
                    policy._vae = ConditionalVAE(
                        obs_dim=ckpt["obs_dim"],
                        act_dim=ckpt["act_dim"],
                        latent_dim=ckpt["latent_dim"]
                    )
                    policy._vae.load_state_dict(ckpt["state_dict"])
                    policy._vae = policy._vae.to(device).eval()
                
                # Denormalize latent prediction before decoding (if normalization was used)
                if getattr(policy, "use_lspo_norm", True) and getattr(policy, "latent_mean", None) is not None:
                    l_mean = policy.latent_mean.to(device)
                    l_std = policy.latent_std.to(device)
                    pred = pred * (l_std + 1e-6) + l_mean

                action_t = policy._vae.decode(pred, obs_t)
                action = action_t.squeeze(0).cpu().numpy()
            else:
                action = pred.squeeze(0).cpu().numpy()

            # Clip to action space bounds
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

            # Check success flag (provided by gymnasium-robotics Adroit envs)
            if info.get("success", False) or info.get("goal_achieved", False):
                episode_success = True

        episode_rewards.append(total_reward)
        successes.append(episode_success)

    env.close()

    rewards_arr = np.array(episode_rewards)
    return {
        "mean_reward":  float(rewards_arr.mean()),
        "std_reward":   float(rewards_arr.std()),
        "min_reward":   float(rewards_arr.min()),
        "max_reward":   float(rewards_arr.max()),
        "success_rate": float(np.mean(successes)),
        "rewards":      episode_rewards,
    }


def evaluate_from_checkpoint(
    ckpt_path: str,
    n_episodes: int = 50,
    env_name: str = "AdroitHandPen-v1",
    seed: int = 0,
    device_str: str = "cpu",
) -> dict[str, float]:
    """Convenience wrapper: load policy from checkpoint and evaluate."""
    policy = load_policy(ckpt_path)
    return evaluate_policy(policy, n_episodes=n_episodes, env_name=env_name,
                           seed=seed, device_str=device_str)
