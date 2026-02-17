"""
Gaussian/Trajectory Filtering for RoboMimic Datasets

This module implements a naive Gaussian filtering approach to curate robomimic datasets.
It computes trajectory-level statistics and filters out trajectories that are outliers
based on a multivariate Gaussian distribution.

The filtering approach:
1. Computes trajectory features (mean action, mean reward, trajectory length, etc.)
2. Fits a multivariate Gaussian to these features
3. Scores each trajectory based on its likelihood under the Gaussian
4. Filters out trajectories with low likelihood (outliers)
"""

import os
import h5py
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm
import json


def extract_trajectory_features(dataset_path, obs_keys=None):
    """
    Extract features from each trajectory in the dataset.
    
    Args:
        dataset_path: Path to the HDF5 dataset file
        obs_keys: Optional list of observation keys to use for feature extraction
        
    Returns:
        features: Array of shape (n_episodes, n_features) containing trajectory features
        episode_info: List of dictionaries with episode metadata
    """
    # Resolve path to absolute (using os.path.expanduser like robomimic does)
    dataset_path = os.path.expanduser(str(dataset_path))
    dataset_path = os.path.abspath(dataset_path)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    features = []
    episode_info = []
    
    with h5py.File(dataset_path, 'r') as f:
        demos = list(f['data'].keys())
        demos = sorted(demos, key=lambda x: int(x.split('_')[1]))
        
        for demo_key in tqdm(demos, desc="Extracting trajectory features"):
            demo = f[f'data/{demo_key}']
            
            # Get actions
            actions = demo['actions'][:]
            n_steps = actions.shape[0]
            
            # Compute trajectory-level statistics
            mean_action = np.mean(actions, axis=0)
            std_action = np.std(actions, axis=0)
            action_magnitude = np.mean(np.linalg.norm(actions, axis=1))
            
            # Get rewards if available
            if 'rewards' in demo:
                rewards = demo['rewards'][:]
                mean_reward = np.mean(rewards)
                total_reward = np.sum(rewards)
            else:
                mean_reward = 0.0
                total_reward = 0.0
            
            # Get states if available (for state-based features)
            if 'states' in demo:
                states = demo['states'][:]
                state_variance = np.mean(np.var(states, axis=0))
            else:
                state_variance = 0.0
            
            # Combine features into a feature vector
            # Features: [mean_action (flattened), std_action (flattened), 
            #           action_magnitude, mean_reward, total_reward, 
            #           trajectory_length, state_variance]
            feature_vec = np.concatenate([
                mean_action.flatten(),
                std_action.flatten(),
                [action_magnitude],
                [mean_reward],
                [total_reward],
                [n_steps],
                [state_variance]
            ])
            
            features.append(feature_vec)
            episode_info.append({
                'demo_key': demo_key,
                'n_steps': n_steps,
                'mean_reward': mean_reward,
                'total_reward': total_reward
            })
    
    return np.array(features), episode_info


def fit_gaussian_and_score(features):
    """
    Fit a multivariate Gaussian to the features and compute likelihood scores.
    
    Args:
        features: Array of shape (n_episodes, n_features)
        
    Returns:
        scores: Array of likelihood scores for each trajectory
        mean: Mean of the fitted Gaussian
        cov: Covariance matrix of the fitted Gaussian
    """
    # Fit multivariate Gaussian
    mean = np.mean(features, axis=0)
    cov = np.cov(features.T)
    
    # Add small regularization to ensure positive definite
    cov += np.eye(cov.shape[0]) * 1e-6
    
    # Compute likelihood for each trajectory
    try:
        mvn = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
        scores = mvn.logpdf(features)
    except:
        # Fallback: use diagonal covariance if full covariance fails
        print("Warning: Using diagonal covariance matrix")
        cov_diag = np.diag(np.diag(cov)) + np.eye(cov.shape[0]) * 1e-6
        mvn = multivariate_normal(mean=mean, cov=cov_diag, allow_singular=True)
        scores = mvn.logpdf(features)
    
    return scores, mean, cov


def filter_trajectories(dataset_path, output_path, filter_ratio=0.1, 
                       method='lowest_likelihood', obs_keys=None, seed=42):
    """
    Filter trajectories from a robomimic dataset based on Gaussian likelihood.
    
    Args:
        dataset_path: Path to input HDF5 dataset
        output_path: Path to save filtered HDF5 dataset
        filter_ratio: Fraction of trajectories to filter out (0.0 to 1.0)
        method: Filtering method ('lowest_likelihood' or 'threshold')
        obs_keys: Optional list of observation keys
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    # Resolve paths to absolute (using os.path.expanduser like robomimic does)
    dataset_path = os.path.expanduser(str(dataset_path))
    dataset_path = os.path.abspath(dataset_path)
    output_path = os.path.expanduser(str(output_path))
    output_path = os.path.abspath(output_path)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if path has a directory component
        os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    print(f"Loading dataset: {dataset_path}")
    print(f"Extracting trajectory features...")
    features, episode_info = extract_trajectory_features(str(dataset_path), obs_keys)
    
    print(f"Fitting Gaussian distribution to {len(features)} trajectories...")
    scores, mean, cov = fit_gaussian_and_score(features)
    
    # Determine which trajectories to keep
    n_episodes = len(scores)
    n_filter = int(n_episodes * filter_ratio)
    
    if method == 'lowest_likelihood':
        # Filter out trajectories with lowest likelihood
        keep_indices = np.argsort(scores)[n_filter:]
        filter_indices = np.argsort(scores)[:n_filter]
    elif method == 'threshold':
        # Filter out trajectories below a threshold
        threshold = np.percentile(scores, filter_ratio * 100)
        keep_indices = np.where(scores >= threshold)[0]
        filter_indices = np.where(scores < threshold)[0]
    else:
        raise ValueError(f"Unknown filtering method: {method}")
    
    keep_indices = np.sort(keep_indices)
    filter_indices = np.sort(filter_indices)
    
    print(f"\nFiltering Statistics:")
    print(f"  Total trajectories: {n_episodes}")
    print(f"  Keeping: {len(keep_indices)} ({100*(1-filter_ratio):.1f}%)")
    print(f"  Filtering out: {len(filter_indices)} ({100*filter_ratio:.1f}%)")
    print(f"  Mean likelihood (kept): {np.mean(scores[keep_indices]):.4f}")
    print(f"  Mean likelihood (filtered): {np.mean(scores[filter_indices]):.4f}")
    
    # Save filtered dataset
    print(f"\nSaving filtered dataset to: {output_path}")
    with h5py.File(dataset_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        # Copy meta group if it exists
        if 'meta' in f_in:
            f_in.copy('meta', f_out)
        
        # Create data group
        data_grp = f_out.create_group('data')
        
        # Copy all attributes from the data group (especially env_args)
        # This is critical for robomimic to read environment metadata
        for attr_key in f_in['data'].attrs:
            data_grp.attrs[attr_key] = f_in['data'].attrs[attr_key]
        
        # Copy kept episodes and preserve all attributes
        new_demo_idx = 0
        total_samples = 0
        for idx in tqdm(keep_indices, desc="Copying episodes"):
            old_demo_key = episode_info[idx]['demo_key']
            new_demo_key = f'demo_{new_demo_idx}'
            
            # Copy episode data (this copies the group structure)
            f_in.copy(f'data/{old_demo_key}', data_grp, name=new_demo_key)
            
            # Ensure num_samples attribute is set correctly
            # This is critical for robomimic dataset loading
            ep_grp = data_grp[new_demo_key]
            if 'num_samples' not in ep_grp.attrs:
                # Calculate from actions if not present
                num_samples = ep_grp['actions'].shape[0]
                ep_grp.attrs['num_samples'] = num_samples
            else:
                num_samples = ep_grp.attrs['num_samples']
            
            total_samples += num_samples
            new_demo_idx += 1
        
        # Update total samples count in data group attributes
        # This follows robomimic's pattern (see convert_robosuite.py)
        if 'total' in data_grp.attrs:
            del data_grp.attrs['total']
        data_grp.attrs['total'] = total_samples
    
    # Save filtering metadata
    metadata = {
        'original_dataset': str(dataset_path),
        'filtered_dataset': str(output_path),
        'filter_ratio': filter_ratio,
        'method': method,
        'n_original': n_episodes,
        'n_filtered': len(keep_indices),
        'n_removed': len(filter_indices),
        'mean_likelihood_kept': float(np.mean(scores[keep_indices])),
        'mean_likelihood_removed': float(np.mean(scores[filter_indices])),
        'kept_indices': [int(i) for i in keep_indices],
        'removed_indices': [int(i) for i in filter_indices],
        'seed': seed
    }
    
    # Save metadata JSON file
    metadata_path = os.path.splitext(output_path)[0] + '_filtering_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Filtering metadata saved to: {metadata_path}")
    print(f"\nDone! Filtered dataset saved to: {output_path}")


def main():
    """
    Command-line interface for trajectory filtering.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Filter robomimic trajectories using Gaussian likelihood'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to input HDF5 dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save filtered HDF5 dataset'
    )
    parser.add_argument(
        '--filter_ratio',
        type=float,
        default=0.1,
        help='Fraction of trajectories to filter out (default: 0.1)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='lowest_likelihood',
        choices=['lowest_likelihood', 'threshold'],
        help='Filtering method (default: lowest_likelihood)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    filter_trajectories(
        dataset_path=args.dataset,
        output_path=args.output,
        filter_ratio=args.filter_ratio,
        method=args.method,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
