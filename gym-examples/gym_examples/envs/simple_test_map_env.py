import numpy as np
import matplotlib
# Try different backends in case TkAgg has issues
try:
    matplotlib.use('TkAgg')  # Try TkAgg first
except Exception:
    try:
        matplotlib.use('Agg')  # Fallback to Agg (no GUI required)
    except Exception:
        pass  # Let matplotlib choose automatically

import matplotlib.pyplot as plt
from map_env_revised import MapEnv
import time
import os
import traceback
import sys
import shutil
import random
from map_logging_loader import MapLoader
from map_renderer import MapRenderer

from datetime import datetime
import json 


# Ensure no global randomness affects our tests
#FIX: why is seed being used here, seed should be a variable that is passed to the env
random.seed(0)
np.random.seed(0)


def save_episode_metrics(uav_pre_fligt_info,
                         uav_post_flight_info,
                         episode: int,
                         seed: int,
                         base_dir: str = '.'):
    """
    Save the pre- and post-flight metric dicts to JSON.
    Creates a subdir named 'YYYY-MM-DD_HH-MM' under base_dir.
    """
    # 1. Compute total number of UAVs
    total_uavs = len(uav_pre_fligt_info)

    # 2. Build timestamped directory name
    now = datetime.now()
    common_dir_name = 'Metrics'
    
    # total_uav_str = f'total_uav{total_uavs}'
    # dir_name = now.strftime('%Y-%m-%d_%H-%M')
    
    save_dir = os.path.join(base_dir, common_dir_name) # total_uav_str, dir_name
    os.makedirs(save_dir, exist_ok=True)

    # 3. Prepare filename and payload
    filename = f'uav_count_{total_uavs}_episode_{episode}_seed_{seed}.json'
    filepath = os.path.join(save_dir, filename)

    payload = {
        'episode': episode,
        'seed': seed,
        'total_uavs': total_uavs,
        'uav_pre_fligt_info': uav_pre_fligt_info,
        'uav_post_flight_info': uav_post_flight_info
    }

    # 4. Write JSON
    with open(filepath, 'w') as f:
        json.dump(payload, f, indent=4)

    print(f"Saved metrics for episode {episode} -> {filepath}")






# Check for ffmpeg installation
def check_ffmpeg():
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("ffmpeg is installed. Version info:")
            version_info = result.stdout.decode('utf-8').split('\n')[0]
            print(version_info)
            return True
        else:
            print("ffmpeg command found but returned error.")
            return False
    except Exception:
        print("ffmpeg not found. Animation saving may be limited.")
        return False

# Monkey patch the reward function to just return 0
def simple_reward(self):
    """Simplified reward function that just returns 0."""
    return 0.0

# Apply the monkey patches
MapEnv._get_reward = simple_reward

def test_map_env_with_random_actions(number_orca_agents, number_uav, number_of_vp, episodes=2, max_steps_per_episode=50, render=True, 
                                     save_animation=True, env_seed=42, episode_seeds=None,
                                     mp4_only=True):
    """
    Test the MapEnv with random actions to verify all components work together.
    Using a simplified reward function that just returns 0.
    
    Args:
        episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
        render: Whether to render the environment during testing
        save_animation: Whether to save animations
        env_seed: Initial seed for environment creation
        episode_seeds: List of seeds for each episode (if None, will generate)
        mp4_only: If True, only save MP4 and skip backup GIF creation
    """
    # Set up seeds - use clearly different values
    if episode_seeds is None:
        # Generate very different seeds for each episode
        episode_seeds = [env_seed + 1000 * (i+1) for i in range(episodes)]

    print(f"Using environment seed: {env_seed}")
    print(f"Episode seeds: {episode_seeds}")
        
    # Create directory for saving test artifacts
    output_dir = "test_results"
    if os.path.exists(output_dir):
        # Clean directory to avoid old files interfering
        print(f"Cleaning {output_dir} directory...")
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error cleaning {file_path}: {e}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for ffmpeg
    has_ffmpeg = check_ffmpeg()
    if not has_ffmpeg and save_animation:
        print("Note: MP4 animations may not work, but we'll try to save GIFs instead.")

    # Track created episode directories in this test run
    created_episode_dirs = []
    
    try:
        # Initialize environment with selected observation space
        print(f"\n=== Creating environment with seed {env_seed} ===")
        env = MapEnv(
            number_of_uav= number_uav,
            num_ORCA_uav=number_orca_agents,
            number_of_vertiport= number_of_vp,
            location_name="Austin, Texas, USA",
            airspace_tag_list=[], #("amenity", "hospital"), ("aeroway", "aerodrome")
            vertiport_tag_list=[('building', 'commercial')],
            max_episode_steps=max_steps_per_episode,
            seed=env_seed,  # Use the specified environment seed
            obs_space_str= "UAM_UAV", #"UAM_UAV", # "LSTM-A2C",
            sorting_criteria= 'closest first',#None, # "closest first",
            render_mode="human" if render else None,
            max_uavs=100, #set these as some hyperparameters 
            max_vertiports=150, #set these as some hyperparameters 
        )
        
        for episode in range(episodes):
            # Use a very different seed for each episode
            episode_seed = episode_seeds[episode]
            print(f"\n===== Starting Episode {episode+1}/{episodes} with seed {episode_seed} =====")
            
            try:
                # Reset environment with episode-specific seed
                print(f"Resetting environment with seed {episode_seed}...")
                obs, info = env.reset(seed=episode_seed)
                # Print initial observations for debugging
                print(f"Initial observation keys: {obs.keys() if isinstance(obs, dict) else 'not a dict'}")

                #EPISODE metrics
                # TODO: 
                # 1. save pre-flight dict as a JSON with episode no, and date
                env._collect_initial_metrics()
                print(env.uav_pre_fligt_info)
                # save to JSON with episode no.

                # Track the episode directory created by the logger
                episode_timestamp = env.logger.timestamp
                episode_dir = f"episode_{episode_timestamp}"
                # Store the timestamp from the logger for later analysis
                created_episode_dirs.append(episode_dir)
                print(f"Created episode directory: {episode_dir}")

                # Debugging to output to text file
                # with open("output.txt", "w") as file:
                #     print(f"\n===== Starting Episode {episode+1}/{episodes} with seed {episode_seed} =====", file=file)
                #     print(f"Resetting environment with seed {episode_seed}...", file=file)
                
                steps = 0
                collision_detected = False
                goal_reached = False
                
                # Run episode
                for step in range(max_steps_per_episode):
                    # Sample random action
                    action = env.action_space.sample()
                    
                    try:
                        # Take step in environment
                        next_obs, reward, terminated, truncated, info = env.step(action)
                        steps += 1
                        
                        # Print occasional status
                        if step % 5 == 0:  # More frequent updates for shorter episodes
                            goal_dist = info.get('distance_to_goal', 'N/A')
                            if not isinstance(goal_dist, str):
                                goal_dist = f"{goal_dist:.2f}"
                            print(f"Step {step}: action = [{action[0]:.2f}, {action[1]:.2f}], "
                                  f"distance to goal = {goal_dist}")
                        
                        # Optionally render
                        if render:
                            try:
                                env.render()
                                time.sleep(0)  # Slightly longer pause for visibility
                            except Exception as e:
                                print(f"Render error: {e}")
                                traceback.print_exc()
                                render = False
                        
                        # Check termination conditions
                        if terminated:
                            print(f"Goal reached at step {step}!")
                            goal_reached = True
                            break
                            
                        if truncated:
                            if 'collision' in info:
                                print(f"Collision detected at step {step}!")
                                collision_detected = True
                            elif 'timeout' in info and info['timeout']:
                                print(f"Episode timeout after {step} steps!")
                            break
                    
                    except Exception as e:
                        print(f"Error during step {step}:")
                        traceback.print_exc()
                        print(f"Continuing to next episode...")
                        break
                
                #TODO: place the metrics extraction code here 
                # 1. collect metrics
                # 2. save metrics dict as JSON with episode no, and date
                # 3. use the saved metrics to plot  
                env._collect_episode_end_metrics()
                print(env.uav_post_flight_info)
                # save to JSON with episode number
                save_episode_metrics(env.uav_pre_fligt_info, env.uav_post_flight_info, episode, episode_seed) 



                # Episode summary
                # print(f"\nEpisode {episode+1} summary:")
                # print(f"Total steps: {steps}")
                # print(f"Goal reached: {goal_reached}")
                # print(f"Collision detected: {collision_detected}")
                
                # Create animation for this episode
                if save_animation and steps > 0:
                    try:
                        print("\nCreating animation...")
                        # Create animation with the full number of steps taken
                        ani = env.create_animation(steps)
                        if ani:
                            save_path = os.path.join(output_dir, f"episode_{episode+1}_seed_{episode_seed}_animation")
                            print(f"Saving animation to {save_path}")
                            env.save_animation(ani, save_path, mp4_only)
                            
                            # Check if animation was successfully saved
                            gif_path = f"{save_path}.gif"
                            mp4_path = f"{save_path}.mp4"

                            if not mp4_only and os.path.exists(gif_path):
                                print(f"GIF successfully saved at: {gif_path}")
                                file_size = os.path.getsize(gif_path) / (1024 * 1024)  # Size in MB
                                print(f"GIF file size: {file_size:.2f} MB")
                            elif not mp4_only:
                                print(f"Warning: GIF file not found at {gif_path}")
                                
                            if os.path.exists(mp4_path):
                                print(f"MP4 successfully saved at: {mp4_path}")
                                file_size = os.path.getsize(mp4_path) / (1024 * 1024)  # Size in MB
                                print(f"MP4 file size: {file_size:.2f} MB")
                            else:
                                print(f"Warning: MP4 file not found at {mp4_path}")
                        else:
                            print("Failed to create animation - no animation object returned")
                    except Exception as e:
                        print(f"Animation error: {e}")
                        traceback.print_exc()
            
            except Exception as e:
                print(f"Error during episode {episode+1}:")
                traceback.print_exc()
                print(f"Continuing to next episode...")
                continue
        
        # Close environment
        env.close()
        print("\nEnvironment closed!")
        
        # Analyze the generated logs from this test
        if created_episode_dirs:
            print("\n=== Analyzing Episodes from Current Test Run ===")
            loader = MapLoader(base_log_dir="logs")
            
            for episode_dir in created_episode_dirs:
                print(f"\n=== Analyzing episode: {episode_dir } ===")
                
                # Check if metadata.json exists before trying to load it
                metadata_path = os.path.join("logs", episode_dir, 'metadata.json')
                if not os.path.exists(metadata_path):
                    print(f"Warning: No metadata.json file found for {episode_dir}")
                    continue
                
                try:
                    # Print episode summary with error handling
                    try:
                        # TODO: place a switch here to toggle on/off
                        loader.print_episode_summary(episode_dir)
                    except Exception as e:
                        print(f"Error printing episode summary: {e}")
                    
                    # Get all agents with error handling
                    try:
                        non_learning_agents = loader.get_non_learning_agents(episode_dir)
                        learning_agents = loader.get_learning_agents(episode_dir)
                        
                        print(f"\nNon-learning agents: {non_learning_agents}")
                        print(f"Learning agents: {learning_agents}")
                        
                        # Print details for each agent
                        for agent_id in non_learning_agents + learning_agents:
                            try:
                                loader.print_agent_details(episode_dir, agent_id)
                            except Exception as e:
                                print(f"Error printing details for agent {agent_id}: {e}")
                    except Exception as e:
                        print(f"Error getting agent lists: {e}")
                except Exception as e:
                    print(f"Error analyzing episode {episode_dir}: {e}")
        else:
            print("\nNo episodes were created during this test run.")
                
        print("\nTest completed!")
    
    except Exception as e:
        print("Fatal error during testing:")
        traceback.print_exc()


if __name__ == "__main__":
    print("Testing MapEnv with random actions and enhanced animation capabilities...")
    
    # Check if we can import necessary libraries
    try:
        import matplotlib.animation
        has_animation = True
    except ImportError:
        has_animation = False
        print("Warning: matplotlib.animation not available, animations will be disabled")
    
    # Define very different seeds to clearly demonstrate seed effect
    env_seed = 17  # Main environment seed
    episode_seeds = [60, 75]  # Completely different seeds for each episode
    
    # Increase to 500 steps to see more movement
    test_map_env_with_random_actions(
        number_uav= 3, # 3
        number_orca_agents= 1, # 1
        number_of_vp=5, # 5
        episodes=1,
        max_steps_per_episode=1200,
        render="human",
        save_animation=True,
        env_seed=env_seed,
        episode_seeds=episode_seeds,
        mp4_only=True  # Set to True to only save MP4 files
    )
    
    print("Test script completed.")