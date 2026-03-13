"""Data loading utilities for D4RL Adroit human-v2 datasets via Minari."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader





DATASET_ID = "D4RL/pen/human-v2"

TASK_TO_DATASET_ID = {
    "pen": "D4RL/pen/human-v2",
    "hammer": "D4RL/hammer/human-v2",
    "relocate": "D4RL/relocate/human-v2",
    "door": "D4RL/door/human-v2",
}

TASK_TO_ENV_NAME = {
    "pen": "AdroitHandPen-v1",
    "hammer": "AdroitHandHammer-v1",
    "relocate": "AdroitHandRelocate-v1",
    "door": "AdroitHandDoor-v1",
}


def normalize_task_name(task: str) -> str:
    """Normalize task input strings to canonical keys.

    Examples: "Pen", "hand-pen", "pen" -> "pen"
    """
    t = task.strip().lower().replace("-", "_")
    aliases = {
        "hand_pen": "pen",
        "hand_hammer": "hammer",
        "hand_relocate": "relocate",
        "hand_door": "door",
    }
    return aliases.get(t, t)


def get_task_spec(task: str) -> dict[str, str]:
    """Return canonical task, Minari dataset id, and eval env name."""
    canonical = normalize_task_name(task)
    if canonical not in TASK_TO_DATASET_ID:
        allowed = ", ".join(sorted(TASK_TO_DATASET_ID.keys()))
        raise ValueError(f"Unknown task '{task}'. Expected one of: {allowed}")
    return {
        "task": canonical,
        "dataset_id": TASK_TO_DATASET_ID[canonical],
        "env_name": TASK_TO_ENV_NAME[canonical],
    }






def load_pen_human(dataset_id: str = DATASET_ID) -> dict[str, np.ndarray]:
    """Download (if needed) and return raw numpy arrays from the Minari dataset.

    Returns
    -------
    dict with keys:
        'observations' : np.ndarray  shape (N, 45)
        'actions'      : np.ndarray  shape (N, 24)
        'rewards'      : np.ndarray  shape (N,)
        'terminals'    : np.ndarray  shape (N,) episode-end flags
        'episode_ends' : list[int]   indices where each episode ends (exclusive)
    """
    import minari  

    dataset = minari.load_dataset(dataset_id, download=True)

    all_obs = []
    all_acts = []
    all_rewards = []
    all_terminals = []
    episode_ends = []
    cursor = 0

    for episode in dataset.iterate_episodes():
        obs = np.asarray(episode.observations, dtype=np.float32)
        acts = np.asarray(episode.actions, dtype=np.float32)
        rewards = np.asarray(episode.rewards, dtype=np.float32)

        
        
        terminations = np.asarray(episode.terminations, dtype=bool)
        truncations = np.asarray(episode.truncations, dtype=bool)
        terminals = np.logical_or(terminations, truncations)

        
        
        T = acts.shape[0]
        obs = obs[:T]
        rewards = rewards[:T]
        terminals = terminals[:T]

        all_obs.append(obs)
        all_acts.append(acts)
        all_rewards.append(rewards)
        all_terminals.append(terminals)
        cursor += T
        episode_ends.append(cursor)

    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_acts, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    terminals = np.concatenate(all_terminals, axis=0)

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals.astype(np.float32),
        "episode_ends": episode_ends,
    }


def load_task_human(task: str = "pen") -> dict[str, np.ndarray]:
    """Load one of the supported Adroit human-v2 tasks by task key."""
    spec = get_task_spec(task)
    return load_pen_human(dataset_id=spec["dataset_id"])






class DemoDataset(Dataset):
    """PyTorch dataset wrapping (state, action) demonstration pairs."""

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        weights: np.ndarray | None = None,
        obs_norm: dict[str, np.ndarray] | None = None,
    ):
        obs = observations.copy()
        if obs_norm is not None:
            obs = (obs - obs_norm["mean"]) / (obs_norm["std"] + 1e-6)

        self.observations = torch.from_numpy(obs).float()
        self.actions = torch.from_numpy(actions).float()
        if weights is not None:
            self.weights = torch.from_numpy(weights).float()
        else:
            self.weights = torch.ones(len(observations), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int):
        return self.observations[idx], self.actions[idx], self.weights[idx]






def make_datasets(
    data: dict[str, np.ndarray],
    train_frac: float = 0.8,
    seed: int = 42,
    weights: np.ndarray | None = None,
    obs_norm: dict[str, np.ndarray] | None = None,
) -> tuple[DemoDataset, DemoDataset, np.ndarray, np.ndarray]:
    """Split raw data into train and validation DemoDatasets.
    ...
    """
    N = len(data["observations"])
    rng = np.random.default_rng(seed)
    indices = rng.permutation(N)

    n_train = int(N * train_frac)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_obs = data["observations"][train_idx]
    train_acts = data["actions"][train_idx]
    val_obs = data["observations"][val_idx]
    val_acts = data["actions"][val_idx]

    train_w = weights[train_idx] if weights is not None else None
    val_w = weights[val_idx] if weights is not None else None

    return (
        DemoDataset(train_obs, train_acts, train_w, obs_norm=obs_norm),
        DemoDataset(val_obs, val_acts, val_w, obs_norm=obs_norm),
        train_idx,
        val_idx,
    )


def make_dataloaders(
    train_dataset: DemoDataset,
    val_dataset: DemoDataset,
    batch_size: int = 256,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Wrap datasets in DataLoaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader
