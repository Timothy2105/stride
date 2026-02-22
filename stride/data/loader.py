"""Data loading utilities for the D4RL pen-human-v2 dataset via Minari."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ---------------------------------------------------------------------------
# Dataset ID
# ---------------------------------------------------------------------------
DATASET_ID = "D4RL/pen/human-v2"


# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------

def load_pen_human(dataset_id: str = DATASET_ID) -> dict[str, np.ndarray]:
    """Download (if needed) and return raw numpy arrays from the Minari dataset.

    Returns
    -------
    dict with keys:
        'observations' : np.ndarray  shape (N, 45)
        'actions'      : np.ndarray  shape (N, 24)
        'episode_ends' : list[int]   indices where each episode ends (exclusive)
    """
    import minari  # lazy import so the rest of the codebase can import this module

    dataset = minari.load_dataset(dataset_id, download=True)

    all_obs = []
    all_acts = []
    episode_ends = []
    cursor = 0

    for episode in dataset.iterate_episodes():
        obs = np.asarray(episode.observations, dtype=np.float32)
        acts = np.asarray(episode.actions, dtype=np.float32)

        # Minari episodes store T+1 observations (including terminal).
        # Align so we have T (obs, action) pairs per episode.
        T = acts.shape[0]
        obs = obs[:T]

        all_obs.append(obs)
        all_acts.append(acts)
        cursor += T
        episode_ends.append(cursor)

    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_acts, axis=0)

    return {
        "observations": observations,
        "actions": actions,
        "episode_ends": episode_ends,
    }


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class DemoDataset(Dataset):
    """PyTorch dataset wrapping (state, action) demonstration pairs."""

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        weights: np.ndarray | None = None,
    ):
        self.observations = torch.from_numpy(observations).float()
        self.actions = torch.from_numpy(actions).float()
        if weights is not None:
            self.weights = torch.from_numpy(weights).float()
        else:
            self.weights = torch.ones(len(observations), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int):
        return self.observations[idx], self.actions[idx], self.weights[idx]


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def make_datasets(
    data: dict[str, np.ndarray],
    train_frac: float = 0.8,
    seed: int = 42,
    weights: np.ndarray | None = None,
) -> tuple[DemoDataset, DemoDataset, np.ndarray, np.ndarray]:
    """Split raw data into train and validation DemoDatasets.

    Parameters
    ----------
    data     : dict returned by load_pen_human()
    train_frac : fraction of data used for training
    seed     : RNG seed for reproducibility
    weights  : optional per-sample influence weights (shape N,)

    Returns
    -------
    train_dataset, val_dataset, train_indices, val_indices
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
        DemoDataset(train_obs, train_acts, train_w),
        DemoDataset(val_obs, val_acts, val_w),
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
