"""
Training script for Vertiport Design Problem using PPO with CustomGNN feature extractor.

This script:
1. Creates the VertiportDesignEnv
2. Sets up PPO with CustomGNN feature extractor
3. Trains the policy with logging callbacks
4. Plots training reward curves
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from vp_design_GNN_RL_sb3_v2 import VertiportDesignEnv, CustomGNN



class RewardLoggerCallback(BaseCallback):
    """
    Custom callback for logging episode rewards and distances during training.
    """
    
    def __init__(self, log_freq: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        
        # Storage for plotting
        self.episode_rewards = []
        self.episode_distances = []
        self.episode_lengths = []
        self.timesteps = []
        
        # Current episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_episode_initial_distance = None
        self.current_episode_final_distance = None
        
    def _on_training_start(self) -> None:
        """Called at the start of training."""
        if self.verbose > 0:
            print("Training started...")
    
    def _on_step(self) -> bool:
        """
        Called at each step. Returns True to continue training.
        """
        # Accumulate reward
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Get distance info from environment
        info = self.locals['infos'][0]
        
        # Track initial distance at start of episode
        if self.current_episode_initial_distance is None:
            if 'initial_total_distance' in info:
                self.current_episode_initial_distance = info['initial_total_distance']
            elif 'previous_total_distance' in info and info['previous_total_distance'] is not None:
                pass  # Will be set from current_total_distance
        
        # Track current distance
        if 'current_total_distance' in info:
            self.current_episode_final_distance = info['current_total_distance']
        
        # Check if episode ended
        done = self.locals['dones'][0]
        
        if done:
            # Log episode stats
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.timesteps.append(self.num_timesteps)
            
            if self.current_episode_final_distance is not None:
                self.episode_distances.append(self.current_episode_final_distance)
            
            if self.verbose > 0 and len(self.episode_rewards) % self.log_freq == 0:
                print(f"Episode {len(self.episode_rewards)}: "
                      f"Reward={self.current_episode_reward:.2f}, "
                      f"Length={self.current_episode_length}, "
                      f"Final Distance={self.current_episode_final_distance:.2f}")
            
            # Reset episode tracking
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.current_episode_initial_distance = None
            self.current_episode_final_distance = None
        
        return True
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.verbose > 0:
            print(f"Training finished. Total episodes: {len(self.episode_rewards)}")


def plot_training_results(callback: RewardLoggerCallback, save_path: str = None):
    """
    Plot training reward and distance curves.
    
    Args:
        callback: RewardLoggerCallback with logged data
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    episodes = np.arange(1, len(callback.episode_rewards) + 1)
    
    # Plot 1: Episode Rewards
    ax1 = axes[0, 0]
    ax1.plot(episodes, callback.episode_rewards, alpha=0.6, label='Episode Reward')
    
    # Add moving average
    if len(callback.episode_rewards) >= 10:
        window = min(50, len(callback.episode_rewards) // 5)
        if window > 1:
            moving_avg = np.convolve(callback.episode_rewards, 
                                      np.ones(window)/window, mode='valid')
            ax1.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, 
                    label=f'Moving Avg (window={window})')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Episode Reward')
    ax1.set_title('Training Reward Over Episodes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final Distance per Episode
    ax2 = axes[0, 1]
    if callback.episode_distances:
        ax2.plot(episodes[:len(callback.episode_distances)], 
                callback.episode_distances, alpha=0.6, label='Final Distance')
        
        # Add moving average for distance
        if len(callback.episode_distances) >= 10:
            window = min(50, len(callback.episode_distances) // 5)
            if window > 1:
                moving_avg = np.convolve(callback.episode_distances, 
                                          np.ones(window)/window, mode='valid')
                ax2.plot(episodes[window-1:len(callback.episode_distances)], 
                        moving_avg, 'r-', linewidth=2, 
                        label=f'Moving Avg (window={window})')
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Final Total Distance')
        ax2.set_title('Final Distance Over Episodes (Lower is Better)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative Reward
    ax3 = axes[1, 0]
    cumulative_reward = np.cumsum(callback.episode_rewards)
    ax3.plot(episodes, cumulative_reward, 'g-', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Cumulative Reward')
    ax3.set_title('Cumulative Reward Over Training')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Episode Length
    ax4 = axes[1, 1]
    ax4.plot(episodes, callback.episode_lengths, alpha=0.6)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Episode Length (Steps)')
    ax4.set_title('Episode Length Over Training')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training plots saved to: {save_path}")
    
    plt.show()
    
    return fig


def train_vertiport_design(
    total_timesteps: int = 50000,
    simulator_step: int = 3,
    episode_length: int = 100,
    algorithm: str = "PPO",
    features_dim: int = 16,
    final_dim: int = 8,
    num_gnn_layers: int = 2,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    save_model: bool = True,
    model_save_path: str = None,
    plot_save_path: str = None,
    verbose: int = 1
):
    """
    Train the vertiport design policy.
    
    Args:
        total_timesteps: Total training timesteps
        simulator_step: Steps to run UAM simulator per design step
        episode_length: Steps per episode in design problem
        algorithm: "PPO" or "A2C"
        features_dim: Hidden dimension for GNN
        final_dim: GNN output dimension before concat
        num_gnn_layers: Number of GAT layers
        learning_rate: Learning rate for optimizer
        n_steps: Number of steps per rollout (PPO)
        batch_size: Minibatch size for PPO
        save_model: Whether to save the trained model
        model_save_path: Path to save model
        plot_save_path: Path to save training plots
        verbose: Verbosity level
    
    Returns:
        model: Trained model
        callback: Callback with training logs
    """
    
    # Create timestamp for unique naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create environment
    print("Creating environment...")
    env = VertiportDesignEnv(
        simulator_step=simulator_step,
        total_episode_timestep=episode_length,
        node_feat_dim=3,
        edge_feat_dim=2
    )
    
    # Wrap with Monitor for additional logging
    log_dir = f"./logs/vertiport_design_{timestamp}/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    
    # Policy kwargs with CustomGNN feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomGNN,
        features_extractor_kwargs=dict(
            features_dim=features_dim,
            final_dim=final_dim,
            num_layers=num_gnn_layers
        ),
    )
    
    # Create model
    print(f"Creating {algorithm} model with CustomGNN feature extractor...")
    
    if algorithm.upper() == "PPO":
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            verbose=verbose,
            tensorboard_log=f"./tensorboard_logs/vertiport_{timestamp}/"
        )
    elif algorithm.upper() == "A2C":
        model = A2C(
            policy="MultiInputPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            n_steps=n_steps,
            verbose=verbose,
            tensorboard_log=f"./tensorboard_logs/vertiport_{timestamp}/"
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'PPO' or 'A2C'.")
    
    # Create callback for logging
    reward_callback = RewardLoggerCallback(log_freq=10, verbose=verbose)
    
    # Train
    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"Episode length: {episode_length} steps")
    print(f"Expected episodes: ~{total_timesteps // episode_length}")
    print("-" * 50)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=reward_callback,
        progress_bar=True
    )
    
    print("-" * 50)
    print("Training complete!")
    
    # Save model
    if save_model:
        if model_save_path is None:
            model_save_path = f"./models/vertiport_design_{algorithm}_{timestamp}"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        print(f"Model saved to: {model_save_path}")
    
    # Plot results
    if plot_save_path is None:
        plot_save_path = f"./plots/training_results_{timestamp}.png"
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    
    plot_training_results(reward_callback, save_path=plot_save_path)
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Total episodes: {len(reward_callback.episode_rewards)}")
    print(f"Average episode reward: {np.mean(reward_callback.episode_rewards):.2f}")
    print(f"Final 10 episodes avg reward: {np.mean(reward_callback.episode_rewards[-10:]):.2f}")
    
    if reward_callback.episode_distances:
        print(f"Initial avg distance: {np.mean(reward_callback.episode_distances[:10]):.2f}")
        print(f"Final avg distance: {np.mean(reward_callback.episode_distances[-10:]):.2f}")
        improvement = (np.mean(reward_callback.episode_distances[:10]) - 
                      np.mean(reward_callback.episode_distances[-10:]))
        print(f"Distance improvement: {improvement:.2f}")
    
    return model, reward_callback


def evaluate_policy(model, env, n_episodes: int = 10, verbose: int = 1):
    """
    Evaluate trained policy.
    
    Args:
        model: Trained SB3 model
        env: Environment to evaluate on
        n_episodes: Number of evaluation episodes
        verbose: Verbosity level
    
    Returns:
        dict: Evaluation statistics
    """
    episode_rewards = []
    episode_distances = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(total_reward)
        if 'current_total_distance' in info:
            episode_distances.append(info['current_total_distance'])
        
        if verbose > 0:
            print(f"Eval Episode {ep+1}: Reward={total_reward:.2f}, "
                  f"Final Distance={info.get('current_total_distance', 'N/A')}")
    
    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_distance': np.mean(episode_distances) if episode_distances else None,
        'std_distance': np.std(episode_distances) if episode_distances else None,
    }
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    if stats['mean_distance']:
        print(f"Mean final distance: {stats['mean_distance']:.2f} ± {stats['std_distance']:.2f}")
    
    return stats


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Training configuration
    CONFIG = {
        'total_timesteps': 3200,      # Total training steps
        'simulator_step': 2,            # UAM simulator steps per design step
        'episode_length': 160,          # Steps per design episode
        'algorithm': 'PPO',             # 'PPO' or 'A2C'
        'features_dim': 16,             # GNN hidden dimension
        'final_dim': 8,                 # GNN output dimension
        'num_gnn_layers': 1,            # Number of GAT layers
        'learning_rate': 3e-4,          # Learning rate
        'n_steps': 16,                # Steps per rollout
        'batch_size': 4,               # Minibatch size
        'verbose': 1                    # Verbosity
    }
    
    # Train the model
    model, callback = train_vertiport_design(**CONFIG)
    
    # Create fresh environment for evaluation
    eval_env = VertiportDesignEnv(
        simulator_step=CONFIG['simulator_step'],
        total_episode_timestep=CONFIG['episode_length'],
        node_feat_dim=3,
        edge_feat_dim=2
    )
    
    # Evaluate trained policy
    eval_stats = evaluate_policy(model, eval_env, n_episodes=3, verbose=1)



    ######
    #     CONFIG = {
    #     'total_timesteps': 30000,      # Total training steps
    #     'simulator_step': 2,            # UAM simulator steps per design step
    #     'episode_length': 10000, #this means there are only 3 total episodes, but each episode is really long         # Steps per design episode
    #     'algorithm': 'PPO',             # 'PPO' or 'A2C'
    #     'features_dim': 16,             # GNN hidden dimension
    #     'final_dim': 8,                 # GNN output dimension
    #     'num_gnn_layers': 1,            # Number of GAT layers
    #     'learning_rate': 3e-4,          # Learning rate
    #     'n_steps': 2048,                # Steps per rollout
    #     'batch_size': 64,               # Minibatch size
    #     'verbose': 1                    # Verbosity
    # }