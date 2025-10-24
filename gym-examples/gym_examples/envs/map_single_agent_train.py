#!/usr/bin/env python
"""
Simple training script for MapEnv with UAM_UAV observation space.
This script focuses on basic training and visualization with minimal complexity.
"""

import os
import sys
import logging
import numpy as np
import matplotlib
# Try different backends in case TkAgg has issues
try:
    matplotlib.use('Agg')  # Fallback to Agg (no GUI required)
except Exception:
    pass  # Let matplotlib choose automatically
import matplotlib.pyplot as plt
import time
from datetime import datetime
import random
import signal
import traceback

# Stable-baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Import your environment and reward function
from map_env_revised import MapEnv
from rewards_utils import _get_reward_simple, _get_reward_only_agent

# Create output directories
os.makedirs("training_logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("animations", exist_ok=True)

# Global variable to track if we're in the process of shutting down
is_shutting_down = False

# Setup signal handlers to save model on interrupt
def signal_handler(sig, frame):
    """Handle interruption signals to save model before exiting"""
    global is_shutting_down
    if is_shutting_down:
        # If we get a second interrupt, exit immediately
        print("\nForced exit - model may not be saved properly.")
        sys.exit(1)
    
    print("\nInterrupted. Saving model and shutting down gracefully...")
    is_shutting_down = True
    
    # The actual saving will happen in the main loop

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Setup logging
def setup_logging():
    """Setup logging with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_logs/training_log_{timestamp}.txt"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(), timestamp # why is timestamp returned?

# Simple callback to track progress
class SimpleTrackingCallback(BaseCallback):
    """Very simple callback to track metrics without excessive plotting"""
    
    def __init__(self, check_freq=10, log_dir='training_logs', verbose=0):
        super(SimpleTrackingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        
        # Metrics
        self.rewards = []
        self.episode_lengths = []
        self.collisions = 0
        self.successes = 0
        self.total_episodes = 0
        
        # Current episode tracking
        self.current_reward = 0
        self.current_length = 0
    
    def _on_step(self):
        """Called after each step"""
        # Check for shutdown signal
        global is_shutting_down
        if is_shutting_down:
            return False  # Stop training
        
        # Update current episode stats
        self.current_reward += self.locals.get("rewards")[0]
        self.current_length += 1
        
        # Check if episode finished
        if self.locals.get("dones")[0]:
            self.total_episodes += 1
            info = self.locals.get("infos")[0]
            
            # Record episode stats
            self.rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            
            # Track success/collision
            if info.get("collision", False):
                self.collisions += 1
            elif not info.get("timeout", False):
                self.successes += 1
            
            # Log every few episodes
            if self.total_episodes % self.check_freq == 0:
                recent_rewards = self.rewards[-self.check_freq:]
                recent_lengths = self.episode_lengths[-self.check_freq:]
                
                # Calculate stats
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                avg_length = np.mean(recent_lengths) if recent_lengths else 0
                success_rate = self.successes / max(1, self.total_episodes)
                collision_rate = self.collisions / max(1, self.total_episodes)
                
                # Log stats
                self.logger.record("reward/avg_reward", avg_reward)
                self.logger.record("episode/avg_length", avg_length)
                self.logger.record("episode/success_rate", success_rate)
                self.logger.record("episode/collision_rate", collision_rate)
                
                # Log to console
                logging.info(f"Episode {self.total_episodes} | "
                            f"Avg reward: {avg_reward:.2f} | "
                            f"Success rate: {success_rate:.2f} | "
                            f"Collision rate: {collision_rate:.2f}")
                
                # Save simple plot every 50 episodes
                if self.total_episodes % 50 == 0 and len(self.rewards) > 0:
                    self._save_simple_plot()
            
            # Reset current episode tracking
            self.current_reward = 0
            self.current_length = 0
        
        return True
    
    def _save_simple_plot(self):
        """Save a simple plot of rewards"""
        if len(self.rewards) < 2:
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot individual rewards as dots
        plt.plot(range(len(self.rewards)), self.rewards, 'b.', alpha=0.3)
        
        # Plot moving average if we have enough data
        if len(self.rewards) >= 5:
            window_size = min(10, len(self.rewards))
            rolling_mean = np.convolve(
                self.rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(self.rewards)), 
                    rolling_mean, 'r-', linewidth=2)
        
        plt.title(f'Training Rewards (Episodes: {self.total_episodes})')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.log_dir}/rewards_{self.total_episodes}.png")
        plt.close()

def create_env(seed=42, max_episode_steps=5000):
    """Create a single instance of the environment"""
    # TODO: MapEnv needs new arguments for UAV_5_intruders, and UAV_UAM
    env = MapEnv(
        number_of_uav=0,  # Fewer UAVs for faster training
        num_ORCA_uav = 0,
        number_of_vertiport=10,
        location_name="Austin, Texas, USA",
        airspace_tag_list=[], #("amenity", "hospital"), ("aeroway", "aerodrome")
        max_episode_steps=max_episode_steps,
        seed=seed,
        obs_space_str="UAM_UAV",
        sorting_criteria="closest first",
        render_mode=None,
        max_uavs=100,
        max_vertiports=150
    )

    env = Monitor(env)

    # Override the reward function
    env._get_reward = lambda: _get_reward_only_agent(env) #_get_reward_simple(env)

    return env

def save_model_checkpoint(model, checkpoint_dir, timestamp):
    """Save a model checkpoint with error handling"""
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = f"{checkpoint_dir}/ppo_map_env_{timestamp}.zip"
        model.save(checkpoint_path)
        logging.info(f"Model saved to {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        traceback.print_exc()
        return None

def create_animation(model_path, seed=101, max_steps=500):
    """Create an animation using the trained model"""
    logging.info("Creating animation for visualization")
    
    try:
        # Load the model
        model = PPO.load(model_path)
        
        # Create env with rendering
        env = MapEnv(
            number_of_uav=0,
            num_ORCA_uav=0,
            number_of_vertiport=10,
            location_name="Austin, Texas, USA", 
            airspace_tag_list=[], #("amenity", "hospital"), ("aeroway", "aerodrome")
            max_episode_steps=max_steps,
            seed=seed,
            obs_space_str="UAM_UAV",
            sorting_criteria="closest first",
            render_mode="human",  # Enable rendering
            max_uavs=4,
            max_vertiports=6
        )

        # Override the reward function
        env._get_reward = lambda: _get_reward_simple(env)
        
        # Run one episode
        obs, _ = env.reset(seed=seed)
        done = False
        steps = 0
        
        # Run until done or max steps
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            print("Inside create animation, STEP")
            print(env.agent.dynamics.is_learning)
            steps += 1
            done = terminated or truncated
        
        # Create animation
        logging.info(f"Creating animation after {steps} steps")
        ani = env.create_animation(steps)
        
        if ani:
            # Determine outcome for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if terminated and not info.get("collision", False):
                status = "success"
            elif info.get("collision", False):
                status = "collision"
            else:
                status = "timeout"
                
            # Save the animation
            animation_path = f"animations/flight_{status}_{steps}steps_{timestamp}"
            env.save_animation(ani, animation_path)
            logging.info(f"Animation saved to {animation_path}")
            return True
        else:
            logging.warning("Failed to create animation object")
            return False
    
    except Exception as e:
        logging.error(f"Animation error: {e}")
        traceback.print_exc()
        return False
    finally:
        if 'env' in locals():
            env.close()

def main(total_timesteps=1_000_000, max_episode_steps=1024*3, seed=42):
    """Main training function with better error handling"""
    # Setup
    logger, timestamp = setup_logging()
    logger.info(f"Starting training with {total_timesteps} timesteps and seed {seed}")
    
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Create environment
    try:
        env = create_env(seed=seed, max_episode_steps=max_episode_steps)
        logger.info(f"Environment created successfully")
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        traceback.print_exc()
        return
    
    # Create the model
    try:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=1024,  #number of step(s), 'env.step()' to run in environment before policy update  #Reduced from 2048 for faster updates
            batch_size=64,
            n_epochs=10, #number of passes over the collected data from env.step() above
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            max_grad_norm=0.5,
            tensorboard_log="./training_logs/"
        )
        logger.info("Model created successfully")
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        traceback.print_exc()
        env.close()
        return
    
    # Create callback
    callback = SimpleTrackingCallback(check_freq=10, log_dir="training_logs")
    
    # Train the model with error handling
    try:
        logger.info("Starting training...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps, # total number of interactions with the env, ie number of times env.step() is called
            callback=callback,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Training error: {e}")
        traceback.print_exc()
    finally:
        # Always try to save the model
        model_path = save_model_checkpoint(model, "models", timestamp)
        env.close()
    
    # If model was saved successfully, try to create an animation
    if model_path and os.path.exists(model_path):
        create_animation(model_path, seed=seed+100, max_steps=max_episode_steps)
    
    logger.info("Script completed")

if __name__ == "__main__":
    # Run with smaller numbers initially
    # TODO: update max_episode_steps to have other_UAV completion as a factor.
    #      If other_UAVs complete then episode ends as well
    # TODO: Change the max_episode_steps to 3750   
    # timesteps = 100000 
    # REMEMBER
    # total_timesteps - number of total env.step() called during training 
    # max_episode_steps - number of steps after which env will call env.reset()
    # max_episode_steps - preferably should be multiple/fraction of n_steps, 1024 * 6 (double 1024 * 3 recommended)

    # Training and inference
    main(total_timesteps=1_000_000, max_episode_steps=1024*3, seed=42)

    # Inference
    # model_path = "" # Change to saved model path
    # if model_path and os.path.exists(model_path):
    #     create_animation(model_path, seed=2, max_steps=1024*3)