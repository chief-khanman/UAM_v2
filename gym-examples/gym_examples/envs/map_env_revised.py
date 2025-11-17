import math
import random
import numpy as np
import time
import pandas as pd
import geopandas as gpd
from typing import Tuple

from uav_v2 import UAV_v2
from auto_uav_v2 import Auto_UAV_v2
from controller_static import StaticController
from controller_non_coop import NonCoopController
from controller_non_coop_smooth import NonCoopControllerSmooth
from simple_controllers import SimpleController
from dynamics_point_mass import PointMassDynamics
from map_sensor import MapSensor
from atc import ATC
from airspace import Airspace
from vertiport import Vertiport
from utils_data_transform import transform_sensor_data, choose_obs_space_constructor, transform_for_uam
from mapped_env_util import *

from shapely import Point
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
from gymnasium import spaces
# from UAV_logger import NonLearningLogger
from map_logger import MapLogger
from map_renderer import MapRenderer

# FOR ORCA/RVO2
from RVO2_sim import RVO2_simulator
from controller_ORCA import ORCA_controller
from dynamics_ORCA import ORCA_Dynamics

class MapEnv(gym.Env):
    """
    Urban Air Mobility environment for flying multiple UAVs with an autonomous agent
    in a real-world map with restricted airspace.
    """


    def __init__(
        self,
        number_of_uav = None, # number of UAV for this simulation 
        num_ORCA_uav = None, # number of ORCA UAV for this simulation 
        number_of_vertiport=None, # number of vertiorts for this simulation 
        location_name="Austin, Texas, USA", # name of city 
        airspace_tag_list=[("amenity", "hospital"), ("aeroway", "aerodrome")], # 
        vertiport_tag_list = [],
        vertiport_tag_str = 'commercial',
        num_vertiport_region = 5,
        n_sample_from_region = 2,
        max_episode_steps=1000, #TODO: max_step_per_episode (number of gym steps per episode)
        number_of_other_agents_observed_for_model=7, # max number of other agents that learning agent tracks for LSTM-A2C model 
        sleep_time=0.005,
        seed=70,
        obs_space_str=None, #! this will be used to choose the constructor 
        sorting_criteria=None, # this needs to be defined for LSTM-A2C model
        render_mode=None,
        max_uavs=8, # this is maximum number of UAVs allowed in env
        max_vertiports=12, # this is maximum number of vertiports allowed in env
        create_uavs_at_random_timestep = True,
        make_uav_at_timestep = 300, #! this argument will become redundant 
        vp_design_problem = False
    ):  
        super().__init__()
        
        # ORCA config
        self.num_ORCA_uav = num_ORCA_uav
        
        # UAV creation at random timestep
        self.create_uavs_at_random_timestep = create_uavs_at_random_timestep
        
        # Environment configuration
        self.number_of_uav = number_of_uav
        self.number_of_vertiport = number_of_vertiport
        self.location_name = location_name
        self.airspace_tag_list = airspace_tag_list
        self.vertiport_tag_list = vertiport_tag_list
        self.vertiport_tag_str = vertiport_tag_str
        self.num_vertiport_region = num_vertiport_region
        self.n_sample_from_region = n_sample_from_region
        self.max_uavs = max_uavs
        self.max_vertiports = max_vertiports
        self.max_number_other_agents_observed = number_of_other_agents_observed_for_model
        self.max_episode_steps = max_episode_steps
        self.sleep_time = sleep_time
        self._seed = seed
        self.obs_space_str = obs_space_str
        self.sorting_criteria = sorting_criteria
        self.render_mode = render_mode
        self.current_time_step = 0
        self.render_history = []  # Store positions for trajectory visualization
        self.uav_radius = 17
        self.NMAC_radius = 200
        self.detection_radius = 500
        self.vp_design_problem = vp_design_problem

        self.make_uav_at_timestep = make_uav_at_timestep

        # Initialize logger for non-learning agents
        self.logger = MapLogger(base_log_dir="logs")
        # Initialize renderer
        self.renderer = MapRenderer(self, render_mode, sleep_time)

        #FIX: cannot pass self.agent here, NOW WHAT
        self.observation_space = choose_obs_space_constructor(
            self.obs_space_str, 
            self.max_number_other_agents_observed
        )

        # Check observation space configuration
        if self.obs_space_str == "LSTM-A2C" and self.sorting_criteria is None:
            raise RuntimeError(
                "env.__init__ needs both obs_space_str and sorting_criteria "
                "for LSTM_A2C model ie sequential observation space"
            )
        elif self.obs_space_str == "UAV_5_intruders" and self.sorting_criteria is None:
            raise RuntimeError(
                "env.__init__ needs both obs_space_str and sorting_criteria "
                "for UAV_5_intruders model ie uav_5_intruders observation space"
            )

        self.action_space = spaces.Box(
            low= np.array([0, -1]),  # [acceleration, heading_change]
            high=np.array([1, 1]),
            shape=(2,),  # Updated shape back to (2,)
            dtype=np.float64,
        )

    def step(self, action):
        # print('step:', self.current_time_step)
        #! why is the UAV detection not used, we only use nmac and collsion 
        #FUTURE: 
        # there needs to be two lists, one for learning_agents, and another for non-learning agents
        # if there is a collision we remove items from respective lists
        # collect information from the removed items
        # and continue until episode ends or agent list is empty(empty agent list means all agents had collision) 
        """
        Execute one time step within the environment.
        
        Args:
            action: Action provided by the learning agent [acceleration, heading_change]
            
        Returns:
            obs: Observation of the learning agent's state
            reward: Reward signal
            terminated: Whether the episode has ended due to goal reached
            truncated: Whether the episode has ended due to collision or timeout
            info: Additional information
        """
        # Update learning agent with provided action
        self.agent.dynamics.update(self.agent, action)

        #Update the odometer of AUTO UAV 
        self.agent.update_odometer()
        
        # Save animation data for all UAVs
        self.renderer.add_data(self.agent)
        
        # Check collision with static objects (restricted airspace)
        static_collision, static_collision_info = self.agent.sensor.get_ra_collision(self.agent)
        if static_collision:
            # print("--- Restricted Airspace COLLISION ---")
            # print(f"Collision with restricted airspace: {static_collision_info}")
            self.logger.record_collision([self.agent.id], 'static')

            # Clean up trajectory data for collided agent
            if hasattr(self, 'trajectory_by_id') and self.agent.id in self.trajectory_by_id:
                del self.trajectory_by_id[self.agent.id]
            #      observation,    reward, truncated, terminated, info
            return self._get_obs(), -100,   False,     True,       {"collision": True, "type": "static"}
            
        # Check NMAC and collision for learning agent with other UAVs
        is_nmac, nmac_list = self.agent.sensor.get_nmac(self.agent)
        if is_nmac:
            # print(f"--- Agent {self.agent.id} NMAC ---")
            # print(f"NMAC detected with {[uav.id for uav in nmac_list]}\n")
            # Log NMAC event with all IDs involved
            nmac_ids = [self.agent.id] + [uav.id for uav in nmac_list]
            self.logger.record_nmac(nmac_ids, self.current_time_step)
        
        is_collision, collision_uav_ids = self.agent.sensor.get_uav_collision(self.agent)
        if is_collision:
            # print("--- COLLISION ---")
            # print(f"Collision detected: {is_collision}, and collision with {collision_uav_ids}\n")
            self.logger.record_collision(collision_uav_ids, 'dynamic')

            # Clean up trajectory data for collided agents
            if hasattr(self, 'trajectory_by_id'):
                for uav_id in collision_uav_ids:
                    if uav_id in self.trajectory_by_id:
                        del self.trajectory_by_id[uav_id]

            self.atc.remove_uavs_by_id(collision_uav_ids)
            if self.atc.get_uav_list() == 0:
                # print("No more UAVs in space")
                #       obs,            reward, terminated, truncated, info
                return self._get_obs(), -100,   False,      True,      {"collision": True, "type": "dynamic"}
        
        
        #### UAV (non-learning) ####

        
        # Now update all non-learning UAVs
        #TODO: Parallelization for detection, nmac, and collision - get_collision/nmac and other operations can be parallelized 
        for uav in self.atc.get_uav_list():
            # print('Len uav list:', len(self.atc.get_uav_list()))
            if not isinstance(uav, Auto_UAV_v2):  # Only process non-learning UAVs
                # Save animation data
                self.renderer.add_data(uav)
                
                # Get non-learning UAV's observations and check for collisions
                observation = uav.get_obs()

                # Check static collisions for non-learning UAVs
                static_collision, static_info = uav.sensor.get_ra_collision(uav)
                if static_collision:
                    self.total_collision_count += 1
                    # # print(f"--- UAV {uav.id} Restricted Airspace COLLISION ---")
                    self.logger.record_collision([uav.id], 'static')

                    # Clean up trajectory for this UAV
                    if hasattr(self, 'trajectory_by_id') and uav.id in self.trajectory_by_id:
                        del self.trajectory_by_id[uav.id]
                    
                    #TODO: need to add logic for, when ORCA agent is removed, that agent is also removed from the RVO2_sim properly
                    self.atc.remove_uavs_by_id([uav.id])
                    continue
                
                # Check NMAC and dynamic collisions
                is_nmac, nmac_list = uav.sensor.get_nmac(uav)
                if is_nmac:
                    # Logic for NMAC count
                    self.total_nmac_count += 1
                    # print(f"--- UAV {uav.id} NMAC ---")
                    # print(f"NMAC detected with {[other_uav.id for other_uav in nmac_list]}\n")
                    # Log NMAC event with all IDs involved
                    nmac_ids = [uav.id] + [other_uav.id for other_uav in nmac_list]
                    #TODO: update record_nmac -> logger will add how many NMACs did UAV have 
                    self.logger.record_nmac(nmac_ids, self.current_time_step)
                
                is_collision, collision_uav_ids = uav.sensor.get_uav_collision(uav)
                if is_collision:
                    # print(f"--- UAV {uav.id} COLLISION ---")
                    self.logger.record_collision(collision_uav_ids, 'dynamic')

                    # Clean up trajectory for collided UAVs
                    if hasattr(self, 'trajectory_by_id'):
                        for coll_id in collision_uav_ids:
                            if coll_id in self.trajectory_by_id:
                                del self.trajectory_by_id[coll_id]

                    #TODO: need to add logic for, when ORCA agent is removed, that agent is also removed from the RVO2_sim properly
                    self.atc.remove_uavs_by_id(collision_uav_ids)
                    continue
                
                # Update non-learning UAV's state
                uav_mission_complete_status = uav.get_mission_status()
                uav.set_mission_complete_status(uav_mission_complete_status)

                # If UAV completed its mission, log it and clean up its trajectory
                if uav_mission_complete_status:
                    self.logger.mark_agent_complete(uav.id)
                    if hasattr(self, 'trajectory_by_id') and uav.id in self.trajectory_by_id:
                        del self.trajectory_by_id[uav.id]
                    #TODO: need to add logic for, when ORCA agent is removed, that agent is also removed from the RVO2_sim properly
                    #self.atc.remove_uavs_by_id([uav.id])
                    continue
                
                # Get and apply non-learning UAV's action based on its controller
                uav_action = uav.get_action(observation=observation)
                uav.dynamics.update(uav, uav_action) #updates the current_position

                #Update the odometer of UAV 
                uav.update_odometer()
                
                # Log the state-action pair for this non-learning UAV
                self.logger.log_non_learning_step(uav.id, observation, uav_action)
        
        
        
        #### START - UAV ORCA (non-learning) ####
        
        self.rvo2_sim.step()

        # Sync RVO2 state back to UAV objects for rendering
        for orca_agent_id_str, rvo2_agent_number in self.rvo2_sim.orca_agent_id_2_RVO2_number.items():
            # Get the mapped UAV object
            uav = self.rvo2_sim.orca_agent_id_to_mapped_agent_obj[orca_agent_id_str]
            
            # Get actual velocity from RVO2 simulator (after collision avoidance)
            velocity = self.rvo2_sim.rvo2_sim.getAgentVelocity(rvo2_agent_number)
            
            # Calculate speed from velocity components
            vx, vy = velocity
            speed = math.sqrt(vx**2 + vy**2)
            
            # Calculate heading from velocity
            if speed > 0.01:  # Avoid division by zero for stationary agents
                heading = math.atan2(vy, vx)
            else:
                heading = uav.current_heading  # Keep previous heading if stationary
            
            # Update UAV attributes for rendering
            uav.current_speed = speed
            uav.current_heading = heading
            
            # Position is already updated in rvo2_sim.step(), but update final_heading
            uav.final_heading = math.atan2(uav.end.y - uav.current_position.y,
                                            uav.end.x - uav.current_position.x)

        #### END - UAV ORCA (non-learning) ####
        
        
        
        
        # Get learning agent's observation, reward, and status
        obs = self._get_obs()
        reward = self._get_reward()

        # Log the full transition for the learning agent
        if hasattr(self, 'previous_obs') and hasattr(self, 'previous_action'):
            self.logger.log_learning_transition(
                self.agent.id,
                self.previous_obs,
                self.previous_action,
                reward,
                obs,  # Current obs becomes next_state
                action  # Current action becomes next_action
            )
        else:
            # First step in episode, can't log transition yet
            pass
            
        # Store current state and action for next transition
        self.previous_obs = obs
        self.previous_action = action
        
        # Get mission status and set termination and truncation flags
        agent_mission_complete = self.agent.get_mission_status()
        terminated = agent_mission_complete
        timeout = self.current_time_step >= self.max_episode_steps
        truncated = timeout

        # If agent completed mission, log it and clean up its trajectory
        if agent_mission_complete:
            self.logger.mark_agent_complete(self.agent.id)
            if hasattr(self, 'trajectory_by_id') and self.agent.id in self.trajectory_by_id:
                del self.trajectory_by_id[self.agent.id]
        
        # Prepare info dictionary
        # info = {
        #     'distance_to_goal': self.agent.current_position.distance(self.agent.end),
        #     'current_step': self.current_time_step,
        #     'timeout': timeout
        # }
        agent_info = self.agent.get_obs()[0] #index 0 -> agent DICT
        info = {
            'current_position':agent_info['current_position'],
            'end':agent_info['end'],
            'distance_to_goal':agent_info['distance_to_goal'],
            'current_speed':agent_info['current_speed'],
            'current_heading':agent_info['current_heading'],
            'final_heading':agent_info['final_heading']
        }
        # Add static object detection info
        ra_data_list = self.agent.sensor.get_ra_detection(self.agent)
        if len(ra_data_list) > 0:
            # At least one restricted area detected
            closest_ra = ra_data_list[0]
            if len(ra_data_list) > 1:
                info['static_detection'] = True
                # Find the closest one if multiple are detected
                closest_ra = sorted(ra_data_list, key=lambda x: x['distance'])[0]
                info['distance_to_restricted'] = closest_ra['distance']
        

        if self.create_uavs_at_random_timestep:    
            #### ADDING UAVS at timesteps ####
            # if self.current_time_step != 0 and self.current_time_step % self.make_uav_at_timestep == 0:
            #!                                                           one way of randomizing the times of UAV creation
            if self.current_time_step != 0 and self.current_time_step % (random.randint(1,10) * 100) == 0:
                for vertiport in self.atc.vertiport_list:
                    start = vertiport
                    end = random.sample(
                        list(filter(lambda x: x is not start, self.atc.vertiport_list)),
                        k=1)[0]
                    
                    # print(start, end)

                    # TODO: check if the ORCA agent is added properly -
                    # since its getting added in the middle of the simulation,
                    # the ORCA agent needs to be part of RVO2 
                    # an agent needs to be created in RVO2
                    # added to RVO2 list
                    # and then stepped in RVO2 sim
                    # and all other tasks related to RVO2 
                    # then it needs to be cleared from the env properly -
                    # deleted from env when mission complets 
                    # deleter from RVO2 when mission completes  
                    # collect num ORCA agents
                    new_uav = UAV_v2(self.orca_controller, 
                                    self.orca_dynamics, 
                                    self.map_sensor, 
                                    radius=self.uav_radius, 
                                    nmac_radius=self.NMAC_radius,
                                    detection_radius=self.detection_radius) 
                    #ATC needs to know about these agents - change this style of open addition
                    self.ORCA_agent_list += [new_uav]
                    self.atc.uav_list += [new_uav]
                    
                    
                    
                    # Assign to UAV
                    self.atc.assign_vertiport_uav(
                        new_uav, 
                        start, 
                        end
                    )

                    new_uav.reset_odometer()

                    # add new_uav to RVO2 sim
                    self.rvo2_sim.addAgent(new_agent=new_uav)


            #### ADDING UAVS at timesteps ####
        
        # Increment time step
        self.current_time_step += 1
        # FYI: unless there is a collision with static or dynamic object, 
        #      obs = self._get_obs()
        # print('from env.step() - Observation:')
        # print(obs)
        return obs, reward, terminated, truncated, info



    def _get_reward(self):
        """
        Calculate the reward for the learning agent with the updated observation format.
        Handles both the goal-seeking behavior and collision avoidance.
        """
        # Base reward from existing function
        reward = 0.0

        # Key reward/punishment parameters
        punishment_existence = -0.1
        uav_punishment_closeness = 0
        ra_punishment_closeness = 0
        heading_efficiency = 0
        progress = 0

        reward += punishment_existence
        
        # Get current state information
        obs = self._get_obs()
        current_distance = self.agent.current_position.distance(self.agent.end)
        
        if self.obs_space_str == 'UAM_UAV':
            # Agent info - extract values from numpy arrays
            current_speed = obs['agent_speed'][0]
            current_heading = obs['agent_current_heading'][0]
            current_deviation = obs['agent_deviation'][0]
            
            # UAV intruder info - extract values from numpy arrays
            uav_intruder = obs['intruder_detected'][0]
            dist_to_uav = obs['distance_to_intruder'][0]
            uav_relative_heading = obs['relative_heading_intruder'][0]
            
            # Intruder position info
            intruder_position_x = obs.get('intruder_position_x', [0])[0]
            intruder_position_y = obs.get('intruder_position_y', [0])[0]
            intruder_speed = obs.get('intruder_speed', [0])[0]
            
            # Create a Point object from coordinates for _get_nmac_state_reward function
            from shapely import Point
            if uav_intruder > 0.5:  # If intruder is detected
                intruder_position = Point(intruder_position_x, intruder_position_y)
            else:
                intruder_position = None
            
            # RA intruder info - extract values from numpy arrays
            ra_intruder = obs['restricted_airspace_detected'][0]
            dist_to_ra = obs['distance_to_restricted_airspace'][0]
            ra_relative_heading = obs['relative_heading_restricted_airspace'][0]
        
        # Progress reward
        if hasattr(self, 'previous_distance') and self.previous_distance is not None:
            progress = self.previous_distance - current_distance
            # Exponential scaling for distance factor to emphasize final approach
            distance_factor = np.exp(-current_distance / 5000)
            reward += progress * 15.0 * (1.0 + distance_factor)
        self.previous_distance = current_distance
        
        # Heading efficiency reward
        ref_heading = math.atan2(
            self.agent.end.y - self.agent.current_position.y,
            self.agent.end.x - self.agent.current_position.x
        )
        heading_diff = ((math.degrees(ref_heading) - math.degrees(self.agent.current_heading) + 180) % 360) - 180
        heading_efficiency = np.cos(np.deg2rad(heading_diff))
        reward += heading_efficiency * 0.5

        # UAV intruder collision avoidance
        if uav_intruder < 0.5:  # No UAV intruder detection
            uav_punishment_closeness = 0
        else:  # UAV intruder detected
            normed_nmac_distance = self.agent.nmac_radius / self.agent.detection_radius
            uav_punishment_closeness = -math.exp(normed_nmac_distance - dist_to_uav * 10.0)
            
            # NMAC-specific state-based rewards
            if dist_to_uav <= self.agent.nmac_radius and intruder_position is not None:
                # NMAC situation detected - apply state-based rewards
                nmac_reward = self._get_nmac_state_reward(intruder_position, intruder_speed, dist_to_uav, uav_relative_heading)
                reward += nmac_reward
                
                # Add reward/penalty for response time
                if hasattr(self, 'nmac_detected_time'):
                    # If we already detected NMAC before
                    response_time = self.current_time_step - self.nmac_detected_time
                    # Penalize longer response times exponentially
                    response_penalty = -0.5 * math.exp(min(response_time / 2.0, 5.0))
                    reward += response_penalty
                else:
                    # First time detecting NMAC
                    self.nmac_detected_time = self.current_time_step
            else:
                # If we're no longer in NMAC but were previously
                if hasattr(self, 'nmac_detected_time'):
                    # Reward for successfully exiting NMAC
                    reward += 2.0
                    # Reset NMAC detection time
                    delattr(self, 'nmac_detected_time')
        
        # RA intruder collision avoidance
        if ra_intruder < 0.5:  # No RA intruder detection
            ra_punishment_closeness = 0
        else:  # RA intruder detected
            normed_nmac_distance = self.agent.nmac_radius / self.agent.detection_radius
            ra_punishment_closeness = -math.exp(normed_nmac_distance - dist_to_ra * 10.0)
        
        # Add proximity penalties to total reward
        reward += uav_punishment_closeness + ra_punishment_closeness
        
        # Add goal reached reward (using mission_complete_distance)
        if current_distance < self.agent.mission_complete_distance:
            reward += 1000.0
        
        return reward
    
    def _get_nmac_state_reward(self, position, speed, distance, relative_heading):
        """
        Calculate reward/penalty based on NMAC state tuple:
        (quadrant, relative_speed, relative_heading)
        """
        # Get the quadrant of the intruder
        intruder_pos = position
        agent_pos = self.agent.current_position
        
        # Determine quadrant (1-4)
        dx = intruder_pos.x - agent_pos.x
        dy = intruder_pos.y - agent_pos.y
        quadrant = 0
        if dx > 0 and dy > 0:
            quadrant = 1
        elif dx < 0 and dy > 0:
            quadrant = 2
        elif dx < 0 and dy < 0:
            quadrant = 3
        else:  # dx > 0 and dy < 0
            quadrant = 4
        
        # Determine relative speed category
        intruder_speed = speed
        agent_speed = self.agent.current_speed
        speed_diff = intruder_speed - agent_speed
        
        if abs(speed_diff) < 0.5:  # Within 0.5 units of speed
            rel_speed_category = "same"
        elif speed_diff > 0:
            rel_speed_category = "faster"
        else:
            rel_speed_category = "slower"
        
        # Discretize relative heading to categories
        # 0-45: same direction
        # 46-135: perpendicular
        # 136-180: opposing direction
        abs_rel_heading = abs(relative_heading)
        if abs_rel_heading <= 45:
            heading_category = "same"
        elif abs_rel_heading <= 135:
            heading_category = "perpendicular"
        else:
            heading_category = "opposing"
        
        # Create the state tuple
        state_tuple = (quadrant, rel_speed_category, heading_category)
        
        # Define rewards/penalties for each state tuple
        state_rewards = {
            # Quadrant 1 (intruder ahead and to the right)
            (1, "same", "same"): -0.5,        # Low risk but monitoring needed
            (1, "same", "perpendicular"): -1.0,  # Medium risk - crossing paths
            (1, "same", "opposing"): -2.0,     # High risk - heading toward each other
            (1, "faster", "same"): -0.8,       # Moderate risk - intruder pulling away but in same direction
            (1, "faster", "perpendicular"): -1.5,  # High risk - fast crossing paths
            (1, "faster", "opposing"): -3.0,    # Severe risk - closing fast head-on
            (1, "slower", "same"): -0.3,       # Low risk - catching up to intruder
            (1, "slower", "perpendicular"): -0.7,  # Moderate risk - crossing paths but slower
            (1, "slower", "opposing"): -1.5,    # High risk - opposing but closing more slowly

            # Quadrant 2 (intruder ahead and to the left)
            (2, "same", "same"): -0.5,        # Low risk but monitoring needed
            (2, "same", "perpendicular"): -1.0,  # Medium risk - crossing paths
            (2, "same", "opposing"): -2.0,     # High risk - heading toward each other
            (2, "faster", "same"): -0.8,       # Moderate risk - intruder pulling away but in same direction
            (2, "faster", "perpendicular"): -1.5,  # High risk - fast crossing paths
            (2, "faster", "opposing"): -3.0,    # Severe risk - closing fast head-on
            (2, "slower", "same"): -0.3,       # Low risk - catching up to intruder
            (2, "slower", "perpendicular"): -0.7,  # Moderate risk - crossing paths but slower
            (2, "slower", "opposing"): -1.5,    # High risk - opposing but closing more slowly

            # Quadrant 3 (intruder behind and to the left)
            (3, "same", "same"): -0.2,        # Very low risk - going same direction behind
            (3, "same", "perpendicular"): -0.7,  # Moderate risk - could intersect paths
            (3, "same", "opposing"): 0.5,      # Low risk - moving away from each other
            (3, "faster", "same"): -1.0,       # Moderate to high risk - intruder catching up
            (3, "faster", "perpendicular"): -1.2,  # High risk - intruder could intercept
            (3, "faster", "opposing"): 0.3,     # Very low risk - fast separation
            (3, "slower", "same"): 0.1,        # Very low risk - pulling away from intruder
            (3, "slower", "perpendicular"): -0.3,  # Low risk - slow crossing behind
            (3, "slower", "opposing"): 0.7,     # Very low risk - both moving apart

            # Quadrant 4 (intruder behind and to the right)
            (4, "same", "same"): -0.2,        # Very low risk - going same direction behind
            (4, "same", "perpendicular"): -0.7,  # Moderate risk - could intersect paths
            (4, "same", "opposing"): 0.5,      # Low risk - moving away from each other
            (4, "faster", "same"): -1.0,       # Moderate to high risk - intruder catching up
            (4, "faster", "perpendicular"): -1.2,  # High risk - intruder could intercept
            (4, "faster", "opposing"): 0.3,     # Very low risk - fast separation
            (4, "slower", "same"): 0.1,        # Very low risk - pulling away from intruder
            (4, "slower", "perpendicular"): -0.3,  # Low risk - slow crossing behind
            (4, "slower", "opposing"): 0.7,     # Very low risk - both moving apart

            # Edge cases - immediate collision risks
            # These are special cases that represent extremely high risk scenarios
            
            # Head-on collision course
            (1, "same", "opposing"): -2.5,     # Direct collision course from front
            (2, "same", "opposing"): -2.5,     # Direct collision course from front
            
            # Fast intruder from behind
            (3, "faster", "same"): -1.5,       # Being overtaken quickly from behind
            (4, "faster", "same"): -1.5,       # Being overtaken quickly from behind
            
            # Perpendicular crossing with high speed differential
            (1, "faster", "perpendicular"): -2.0,  # Fast perpendicular crossing from front-right
            (2, "faster", "perpendicular"): -2.0,  # Fast perpendicular crossing from front-left
            
            # Special case: imminent collision (assuming this can be detected)
            # This could be identified by additional logic checking time-to-collision
            # For very small distances combined with certain headings
        }
        
        # Default penalty if state not explicitly defined
        default_penalty = -1.0
        
        # Get the reward from the dictionary, or use default penalty
        reward = state_rewards.get(state_tuple, default_penalty)
        
        # Additional scaling based on distance (more severe as distance decreases)
        # Normalize distance within NMAC radius (0 to 1)
        norm_distance = max(0.01, min(1.0, distance / self.agent.nmac_radius))
        distance_factor = 1.0 / norm_distance
        
        # Scale reward by distance factor (limited to avoid extreme values)
        distance_scaling = min(3.0, distance_factor)
        reward = reward * distance_scaling
        
        return reward

    def _get_obs(self):

        #FIX: 
        # Depending on the type of constructor used,
        # this method will need to return the correct format of obs data
        """Returns observation of the agent in a specific format (sequential or graph)"""
        if self.obs_space_str == "LSTM-A2C":
            # Get Auto-UAV observation data
            raw_obs = self.agent.get_obs()

            #!DEBUG print statement
            # with open("output.txt", "w") as file:
            #     print(f"Raw observation format: {type(raw_obs)}", file=file)
            #     print(f"Raw observation content: {raw_obs}", file=file)

            # Transform data for sequential observation format
            transformed_data = transform_sensor_data(
                raw_obs,
                self.max_number_other_agents_observed,
                self.obs_space_str,
                self.sorting_criteria,
            )
            
            return transformed_data
            
        elif self.obs_space_str == "GNN-A2C":
            # Get Auto-UAV observation data
            raw_obs = self.agent.get_obs()
            
            # Transform data for graph observation format
            transformed_data = transform_sensor_data(
                raw_obs, 
                self.max_number_other_agents_observed, 
                self.obs_space_str
            )
            
            return transformed_data
        elif self.obs_space_str == 'UAM_UAV':
            #FIX: test this - chech if this method works and produces correct response
        #   (own_dict, (other_agents, restricted_areas))
            raw_obs = self.agent.get_obs()
            #!DEBUG print statement
            # print('from env._get_obs()')
            # print('raw obs:')
            # print(raw_obs)
            # with open("output.txt", "w") as file:
            #     print(raw_obs, file=file)
            # print('raw obs before:')
            # print(raw_obs)
            # Transform data for UAM observation format
            transformed_data = transform_sensor_data(
                raw_obs,
                self.max_number_other_agents_observed,
                'UAM_UAV'
            )
            # print('transformed obs:')
            # print(transformed_data)

            # print('checking raw obs:')
            # print(raw_obs)
            return transformed_data
        
        elif self.obs_space_str == 'UAV_5_intruders':
            # Get Auto-UAV observation data
            raw_obs = self.agent.get_obs()
            #!DEBUG print statement
            # print('from env._get_obs()')
            # print('raw obs:')
            # print(raw_obs)
            # with open("output.txt", "w") as file:
            #     print(f"Raw observation format: {type(raw_obs)}", file=file)
            #     print(f"Raw observation content: {raw_obs}", file=file)

            # Transform data for sequential observation format
            transformed_data = transform_sensor_data(
                raw_obs,
                self.max_number_other_agents_observed,
                self.obs_space_str,
                self.sorting_criteria,
            )
            
            return transformed_data

        else:
            raise RuntimeError(
                "_get_obs \n incorrect self.obs_space_str, check __init__ and self.obs_space_str"
            )


    def set_airspace_vp_design(self):
        '''Make airspace for vp design problem'''
        self.airspace = Airspace(
            self.number_of_vertiport, 
            self.location_name, 
            airspace_tag_list=self.airspace_tag_list, 
            vertiport_tag_list=self.vertiport_tag_list
            )
        
        return None



    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        # Reset internal state
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            random.seed(self._seed)
            np.random.seed(self._seed)
            # print(f"Environment reset with seed: {self._seed}")
            
        # Reset logger and rendering for new episode
        self.logger.reset()
        self.renderer.reset()

        # Reset tracking for learning AGENT logging
        self.previous_obs = None
        self.previous_action = None

        # Reset environment state
        self.current_time_step = 0
        
        # Create the airspace with restricted areas
        if not self.vp_design_problem:
            self.airspace = Airspace(self.number_of_vertiport, self.location_name, airspace_tag_list=self.airspace_tag_list, vertiport_tag_list=self.vertiport_tag_list)
        
        # Create space, sensors, controllers, and dynamics
        self.atc = ATC(airspace=self.airspace, seed=self._seed)
        
        self.map_sensor = MapSensor(airspace=self.airspace, atc=self.atc)
        self.static_controller = StaticController(0, 0)
        self.non_coop_smooth_controller = NonCoopControllerSmooth(10, 2)
        self.non_coop_controller = NonCoopController(10, 1)
        self.pm_dynamics = PointMassDynamics()
        self.agent_pm_dynamics = PointMassDynamics()

        # ORCA dynamics and controller 
        self.orca_controller = ORCA_controller(12,12)
        self.orca_dynamics = ORCA_Dynamics()

        #TODO: UPDATE HOW VERTIPORTS ARE CREATED 
        # Create vertiports
        #TODO: this line will cut vertiports and reduce them to 10 from 16 
        print(f'Before minimizing vertiports: {self.number_of_vertiport}')
        num_vertiports = min(self.max_vertiports, self.number_of_vertiport)  # Use a reasonable number
        #! LETS print and make sure we are actually using all the vertiports from the vertiport/region builder 
        #! THERE should be 16 vertiports 4 regions 
        print(num_vertiports)
        #! WHY is this variable not used



        # if not self.vp_design_problem:
        #     #* create vertiport OPTION 1
        #     # self.airspace.create_n_random_vertiports(num_vertiports, seed=self._seed)
        #     # print('Vertiports in atc-airspace: ',self.atc.vertiport_list)
        #     # print('Vertiports in airspace: ', self.airspace.vertiport_list)
            
        #     #* create vertiport OPTION 2
        #     self.airspace.create_vertiports_from_regions(self.vertiport_tag_str, self.num_vertiport_region, self.n_sample_from_region)
        #     # print('Vertiports in atc-airspace: ',self.atc.vertiport_list)
        #     # print('Vertiports in airspace: ', self.airspace.vertiport_list)
        #     # time.sleep(10)

        if not self.vp_design_problem:
            if self.vertiport_tag_list:  # Only use regions if tags provided
                self.airspace.create_vertiports_from_regions(self.vertiport_tag_str, self.num_vertiport_region, self.n_sample_from_region)
            else:  # Fall back to random placement
                num_vertiports = min(self.max_vertiports, self.number_of_vertiport)
                self.airspace.create_n_random_vertiports(num_vertiports, seed=self._seed)
        else: 
            print('MapEnv is set for Vertiport Design Problem')
            print(f'from map_env_revised.reset() -> Vertiport List: {self.airspace.vertiport_list}')

        # Verify we have at least 2 vertiports
        if not self.vp_design_problem and len(self.airspace.get_vertiport_list()) < 2:
            raise RuntimeError("Failed to create at least 2 vertiports - cannot assign meaningful start/end points")
        
        # Create learning agent first to ensure it gets priority in vertiport assignment
        self.agent = Auto_UAV_v2(
            dynamics=self.agent_pm_dynamics,
            sensor=self.map_sensor,
            radius=self.uav_radius,
            nmac_radius=self.NMAC_radius,
            detection_radius=self.detection_radius,
        )
        if self.vp_design_problem:
            self.agent.controller = SimpleController(self.agent.max_acceleration, self.agent.max_heading_change)

        #### START - Create ORCA agents ####
        
        # make list of ORCA agents
        self.ORCA_agent_list = [UAV_v2(self.orca_controller, self.orca_dynamics, self.map_sensor, radius=self.uav_radius, nmac_radius=self.NMAC_radius,detection_radius=self.detection_radius) for _ in range(self.num_ORCA_uav)]
        # add ORCA controller UAV to uav_list
        self.atc.uav_list += self.ORCA_agent_list
        #TODO: these agents needs to work with vertiport assignment
        #TODO: these agents need to check when they leave and reach vertiport

        self.rvo2_sim = RVO2_simulator(timestep=0.1, 
                                       radius=self.uav_radius, 
                                       max_speed=80, 
                                       mapped_env_orca_agent_list=self.ORCA_agent_list) 
        # collect Restricted Airspace polygons
        if hasattr(self.airspace, 'restricted_airspace_buffer_geo_series'):
            self.rvo2_sim.set_polygon_coords(self.airspace.restricted_airspace_buffer_geo_series)



        #### END --- ORCA agents ####
        
        # Set learning agent in space
        self.atc._set_uav(self.agent)
        
        # Assign agent to vertiports
        vertiports = self.airspace.get_vertiport_list()
        # print('Printing: In file map_env_revised')
        # print(vertiports)
        start_idx = 0
        end_idx = 1 % len(vertiports) # Ensure different start and end
        
        self.atc.assign_vertiport_uav(
            self.agent, 
            vertiports[start_idx], 
            vertiports[end_idx]
        )

        # Reset odometer for AutoUAV
        self.agent.reset_odometer()

        # Create non-learning UAVs
        num_uavs = min(self.max_uavs, self.number_of_uav)

        # Set non-learning UAVs in space and assign to vertiports
        for _ in range(num_uavs):
            self.atc.create_uav(
                UAV_v2,
                controller=self.non_coop_smooth_controller,
                dynamics=self.pm_dynamics,
                sensor=self.map_sensor,
                radius=self.uav_radius,
                nmac_radius=self.NMAC_radius,
                detection_radius=self.detection_radius,
            )

        # Assign start and end points for non-learning UAVs - with controlled randomness
        used_start_indices = [start_idx]  # Track used start indices
        used_end_indices = [end_idx]      # Track used end indices
        
        for i, uav in enumerate(self.atc.get_uav_list()):
            if isinstance(uav, Auto_UAV_v2):
                continue  # Skip the learning agent which was already assigned
                
            # Find unused vertiports if possible
            available_starts = [i for i in range(len(vertiports)) if i not in used_start_indices]
            if not available_starts:  # If all are used, then allow reuse
                available_starts = list(range(len(vertiports)))
                
            # Select start vertiport
            uav_start_idx = random.choice(available_starts)
            used_start_indices.append(uav_start_idx)
            
            # Find end vertiport different from start
            available_ends = [i for i in range(len(vertiports)) if i != uav_start_idx]
            if not available_ends:
                available_ends = [(uav_start_idx + 1) % len(vertiports)]
                
            uav_end_idx = random.choice(available_ends)
            used_end_indices.append(uav_end_idx)
            
            # Assign to UAV
            self.atc.assign_vertiport_uav(
                uav, 
                vertiports[uav_start_idx], 
                vertiports[uav_end_idx]
            )

            #Reset Odometer
            uav.reset_odometer() 
        
        
        # Start/Reset total_nmac_count and collision_count
        #TODO: move this attribute to UAV base class 
        self.total_nmac_count = 0
        self.total_collision_count = 0
        # Initialize tracking for reward function
        self.previous_distance = self.agent.current_position.distance(self.agent.end)

        # Get initial observation and info
        obs = self._get_obs()

        agent_info = self.agent.get_obs()[0]

        info = {
            'current_position':agent_info['current_position'],
            'end':agent_info['end'],
            'distance_to_goal':agent_info['distance_to_goal'],
            'current_speed':agent_info['current_speed'],
            'current_heading':agent_info['current_heading'],
            'final_heading':agent_info['final_heading']
        }

        # # print debug info for vertiport assignments
        # print("--- Vertiport Assignments ---")
        for uav in self.atc.get_uav_list():
            if not isinstance(uav, Auto_UAV_v2):  # Only # print for non-learning UAVs
                # print(f'UAV {uav.id} - Start: {uav.start} end: {uav.end}')
                pass
        # print(f'Agent {self.agent.id} - Start: {self.agent.start} end: {self.agent.end}')
        # print("---------------------------")
        
        #### START - RESET ORCA/RVO2 ####
        self.rvo2_sim.reset()
        #### END - RESET ORCA/RVO2 ####
        
        #ADDING data to renderer
        self.renderer.add_data(self.agent)
        for uav in self.atc.get_uav_list():
            self.renderer.add_data(uav)

        return obs, info
    
    def render(self):
        """Render the current state of the environment."""
        return self.renderer.render()
    
    def create_animation(self, env_time_step):
        """Create an animation of the environment."""
        return self.renderer.create_animation(env_time_step)
    
    def save_animation(self, animation_obj, file_name, mp4_only=True):
        """Save animation to file."""
        return self.renderer.save_animation(animation_obj, file_name, mp4_only)

    def close(self):
        """Close the environment and clean up resources."""
        # Close renderer
        self.renderer.close()

        # Close the logger
        self.logger.close()





        """Old goal direction dense reward function with punishments for static and dynamic collision avoidance.
    
    Reward worked for goal direction task but failed to conduct collision avoidance
    Trained on 25,000 steps with PPO"""
    
    def _collect_uav_initial_metrics(self, ):
        '''For all UAVs in the env collect their 
        distance: start-end point, 
        time_avg: dist/avg-speed,
        arrival_rate,
        service-rate '''

        self.uav_pre_fligt_info = {}
        
        for uav in self.atc.get_uav_list():
            self.uav_pre_fligt_info[uav.id] = {'distance': uav.start.distance(uav.end), #edge attr
                                          'time_avg': uav.start.distance(uav.end)/uav.max_speed, #edge attr
                                          'arrival_rate': 0,  # this info will come from start vertiport - node attr
                                          'service_rate': 0 } # this is an expected service rate of UAVs - node attr}
            
        
        
    
    def _collect_uav_episode_end_metrics(self):
        '''For all UAVs in the env collect the following
         distance-factor: dist-covered(odometer reading)/distance,
         time-factor: time2reach/time-avg,
         NMAC count,
         RA violation count,'''
        

        self.uav_post_flight_info = {}
        for uav in self.atc.get_uav_list():
            
            self.uav_post_flight_info[uav.id] = {
                                          'distance_factor': uav.odometer_reading/uav.start.distance(uav.end), #! node attr, WHICH VERTIPORT SHOULD HOLD THIS ATTR #TODO: what if this metric was multiplied with percentage mission_completion
                                          #'time_factor': time_traveled/self.uav_post_fligt_info[uav.id]['distance']/uav.max_speed,
                                          'NMAC_count': uav.nmac_count,#! node attr, WHICH VERTIPORT SHOULD HOLD THIS ATTR
                                          'TOTAL_NMAC_count': self.total_nmac_count,  #! node attr, WHICH VERTIPORT SHOULD HOLD THIS ATTR# this is total nmac incidence during episode, ie for all UAVs, might need to change it to individual UAV
                                          'RA_violation_count': self.total_collision_count,#! node attr, WHICH VERTIPORT SHOULD HOLD THIS ATTR
                                          'mission_complete': uav.mission_complete_status} #! node attr, WHICH VERTIPORT SHOULD HOLD THIS ATTR# this is total RA collision incidence during episode, ie for all UAVs, might need to change it to individual UAV

        return None
    



    def _collect_pre_flight_vertiport_metrics(self):
        vp_metrics_dict = {}
        for vertiport in self.airspace.vertiport_list:
            vp_metrics_dict[vertiport] = vertiport.metrics
    
    
    # START - General UAV creation method during step()
    #TODO: in step(), remove the control flow and UAV build logic and replace with general methods below
    # start vertiport
    # end vertiport
    # UAV_v2 (controller, dynamics, sensor, radius, nmac, detection)
    
    # Depending on controller
    #   agent_list - ORCA_agent_list
    # atc.uav_list
    # atc.assign_vertiport_uav()
    # new_uav.reset_odometer 
    def _add_uav(self, uav_str:str, start:Vertiport, end:Vertiport) -> None:
        '''From the following list - 
        simple, 
        ORCA,
        [future categories of UAV]  use the string to create a UAV '''
        if uav_str == 'simple':
            raise NotImplementedError('The following case has not been implemented yet')
        elif uav_str == 'ORCA':
            #! for loop - will create a UAV at each vertiport in simulation airspace 
            for vertiport in self.atc.vertiport_list:
                start = vertiport
                end = random.sample(
                    list(filter(lambda x: x is not start, self.atc.vertiport_list)),
                    k=1)[0]
                
                # print(start, end)

                # TODO: check if the ORCA agent is added properly -
                # since its getting added in the middle of the simulation,
                # the ORCA agent needs to be part of RVO2 
                # an agent needs to be created in RVO2
                # added to RVO2 list
                # and then stepped in RVO2 sim
                # and all other tasks related to RVO2 
                # then it needs to be cleared from the env properly -
                # deleted from env when mission complets 
                # deleter from RVO2 when mission completes  
                # collect num ORCA agents
                new_uav = UAV_v2(self.orca_controller, 
                                 self.orca_dynamics, 
                                 self.map_sensor, 
                                 radius=self.uav_radius, 
                                 nmac_radius=self.NMAC_radius,
                                 detection_radius=self.detection_radius) 
                #ATC needs to know about these agents - change this style of open addition
                self.ORCA_agent_list += [new_uav]
                self.atc.uav_list += [new_uav]
                
                

                # Assign to UAV
                self.atc.assign_vertiport_uav(
                    new_uav, 
                    start, 
                    end
                )

                new_uav.reset_odometer()

                # add new_uav to RVO2 sim
                self.rvo2_sim.addAgent(new_agent=new_uav)
                print('New ORCA agent has been added to simulation')

        else:
            raise RuntimeError('Incorrect uav str for uav creation')
        pass

    
    def _get_start_end_vertiport(self,) -> Tuple[Vertiport]:
        '''Provide a start and end vertiport for UAVs'''
        # access a vertiport - 
        # check its queue 
        # if queue not full add to possible_start_vertiport_list 
        # from possible_start_vertiport_list randomly select one vertiport as start vertiport 
        # randomly choose end vertiport 
        start = None
        end = None
        return start, end
        
    # END   - General UAV creation method during step()
    
    
    
    
    
    # def _get_reward(self):
    #     #FIX: 
    #     # Depending on the type of obs_constructor used,
    #     # this method will need to choose the correct reward function
    #     # reward functions should be defined in mapped_env_utils 

    #     # """Calculate the reward for the learning agent."""
    #     reward = 0.0

    #     # Variables
    #     punishment_existence = -0.1
    #     uav_punishment_closeness = 0
    #     ra_punishment_closeness = 0
    #     heading_efficiency = 0
    #     progress = 0


    #     reward += punishment_existence
        
    #     # Get current state information
    #     obs = self._get_obs
    #     current_distance = self.agent.current_position.distance(self.agent.end)
    #     if self.obs_space_str == 'UAM_UAV':
    #         # Agent info
    #         current_speed = obs['agent_speed']
    #         current_heading = obs['agent_current_heading']
    #         current_deviation = obs['agent_deviation']
    #         # UAV intruder info
    #         uav_intruder = obs['intruder_detected']
    #         dist_to_uav = obs['distance_to_intruder']
    #         uav_relative_heading = obs['relative_heading_intruder']
    #         # RA intruder info
    #         ra_intruder = obs['restricted_airspace_detected']
    #         dist_to_ra = obs['distance_to_restricted_airspace']
    #         ra_relative_heading = obs['relative_heading_restricted_airspace']
        
    #     # Progress reward
    #     if hasattr(self, 'previous_distance') and self.previous_distance is not None:
    #         progress = self.previous_distance - current_distance
    #         # Exponential scaling for distance factor to emphasize final approach
    #         distance_factor = np.exp(-current_distance / 5000)
    #         reward += progress * 15.0 * (1.0 + distance_factor)
    #     self.previous_distance = current_distance
        
    #     # Heading efficiency reward
    #     ref_heading = math.atan2(
    #         self.agent.end.y - self.agent.current_position.y,
    #         self.agent.end.x - self.agent.current_position.x
    #     )
    #     heading_diff = ((math.degrees(ref_heading) - math.degrees(self.agent.current_heading) + 180) % 360) - 180
    #     heading_efficiency = np.cos(np.deg2rad(heading_diff))

    #     # UAV intruder collision avoidance
    #     # No UAV intruder detection
    #     if uav_intruder == 0:
    #         uav_punishment_closeness = 0
    #     # UAV intruder detected 
    #     else:
    #         normed_namc_distance = self.agent.nmac_radius / self.agent.detection_radius
    #         uav_punishment_closeness = -math.exp(normed_namc_distance - dist_to_uav * 10.0)
        
    #     # RA intruder collision avoidance
    #     # No RA intruder detection
    #     if ra_intruder == 0:
    #         ra_punishment_closeness = 0
    #     # RA intruder detected
    #     else:
    #         #!TODO mess around with constants to put notion of relative importance for UAV versus RA collision avoidance
    #         normed_namc_distance = self.agent.namc_radius / self.agent.detection_radius
    #         ra_punishment_closeness = -math.exp(normed_namc_distance - dist_to_ra * 10.0)
        
    #     # # Static collision avoidance rewards
    #     # static_detection, static_info = self.agent.sensor.get_static_detection(self.agent)
    #     # if static_detection:
    #     #     # Strong penalty for being near restricted airspace
    #     #     distance_to_restricted = static_info.get('distance', float('inf'))
    #     #     distance_factor = 1.0 - min(1.0, distance_to_restricted / self.agent.detection_radius)
    #     #     reward -= distance_factor * 50.0
            
    #     #     # Reduce heading importance during avoidance
    #     #     if abs(heading_diff) < 30.0:
    #     #         reward += heading_efficiency * 1.0  # Reduced reward during avoidance
    #     # else:
    #     #     # Normal heading rewards when no threats
    #     #     if abs(heading_diff) < 5.0:
    #     #         reward += 5.0
    #     #     elif abs(heading_diff) < 30.0:
    #     #         reward += heading_efficiency * 3.0
    #     #     elif abs(heading_diff) > 90.0:
    #     #         reward -= 10.0
        
    #     # # Dynamic collision avoidance rewards
    #     # is_nmac, nmac_list = self.agent.sensor.get_nmac(self.agent)
    #     # if is_nmac:
    #     #     for nmac_uav in nmac_list:
    #     #         distance = self.agent.current_position.distance(nmac_uav.current_position)
    #     #         nmac_factor = 1.0 - (distance / self.agent.nmac_radius)
    #     #         reward -= nmac_factor * 30.0
        
    #     ### We're keeping this for later
    #     # # Speed management reward
    #     # target_speed = self.agent.dynamics.max_speed
    #     # if current_distance < 1000:
    #     #     # Reduce target speed when approaching goal
    #     #     target_speed = max(5.0, self.agent.dynamics.max_speed * (current_distance / 1000))
    #     # speed_efficiency = 1.0 - abs(current_speed - target_speed) / self.agent.dynamics.max_speed
    #     # reward += speed_efficiency * 2.0
        
    #     # Terminal rewards
    #     if current_distance < self.agent.mission_complete_distance:
    #         reward += 1000.0
        
    #     return float(reward)
    
    """Revised reward"""
    # def _get_reward(self):
    #     # Base reward from existing function
    #     reward = 0.0

    #     # Key reward/punishment parameters
    #     punishment_existence = -0.1
    #     uav_punishment_closeness = 0
    #     ra_punishment_closeness = 0
    #     heading_efficiency = 0
    #     progress = 0

    #     reward += punishment_existence
        
    #     # Get current state information
    #     obs = self._get_obs()
    #     current_distance = self.agent.current_position.distance(self.agent.end)
        
    #     if self.obs_space_str == 'UAM_UAV':
    #         # Agent info
    #         current_speed = obs['agent_speed']
    #         current_heading = obs['agent_current_heading']
    #         current_deviation = obs['agent_deviation']
    #         # UAV intruder info
    #         uav_intruder = obs['intruder_detected']
    #         uav_intruder_position = obs['intruder_position']
    #         uav_intruder_speed = obs['intruder_speed']
    #         dist_to_uav = obs['distance_to_intruder']
    #         uav_relative_heading = obs['relative_heading_intruder']
    #         # RA intruder info
    #         ra_intruder = obs['restricted_airspace_detected']
    #         dist_to_ra = obs['distance_to_restricted_airspace']
    #         ra_relative_heading = obs['relative_heading_restricted_airspace']
        
    #     # Progress reward
    #     if hasattr(self, 'previous_distance') and self.previous_distance is not None:
    #         progress = self.previous_distance - current_distance
    #         # Exponential scaling for distance factor to emphasize final approach
    #         distance_factor = np.exp(-current_distance / 5000)
    #         reward += progress * 15.0 * (1.0 + distance_factor)
    #     self.previous_distance = current_distance
        
    #     # Heading efficiency reward
    #     ref_heading = math.atan2(
    #         self.agent.end.y - self.agent.current_position.y,
    #         self.agent.end.x - self.agent.current_position.x
    #     )
    #     heading_diff = ((math.degrees(ref_heading) - math.degrees(self.agent.current_heading) + 180) % 360) - 180
    #     heading_efficiency = np.cos(np.deg2rad(heading_diff))
    #     reward += heading_efficiency * 0.5  # Added this line which was missing

    #     # UAV intruder collision avoidance
    #     # No UAV intruder detection
    #     if uav_intruder == 0:
    #         uav_punishment_closeness = 0
    #     # UAV intruder detected 
    #     else:
    #         normed_nmac_distance = self.agent.nmac_radius / self.agent.detection_radius
    #         uav_punishment_closeness = -math.exp(normed_nmac_distance - dist_to_uav * 10.0)
            
    #         # NEW: NMAC-specific state-based rewards
    #         if dist_to_uav <= self.agent.nmac_radius:
    #             # NMAC situation detected - apply state-based rewards
    #             nmac_reward = self._get_nmac_state_reward(uav_intruder_position, uav_intruder_speed, dist_to_uav, uav_relative_heading)
    #             reward += nmac_reward
                
    #             # Add reward/penalty for response time
    #             if hasattr(self, 'nmac_detected_time'):
    #                 # If we already detected NMAC before
    #                 response_time = self.current_time_step - self.nmac_detected_time
    #                 # Penalize longer response times exponentially
    #                 response_penalty = -0.5 * math.exp(min(response_time / 2.0, 5.0))
    #                 reward += response_penalty
    #             else:
    #                 # First time detecting NMAC
    #                 self.nmac_detected_time = self.current_time_step
    #         else:
    #             # If we're no longer in NMAC but were previously
    #             if hasattr(self, 'nmac_detected_time'):
    #                 # Reward for successfully exiting NMAC
    #                 reward += 2.0
    #                 # Reset NMAC detection time
    #                 delattr(self, 'nmac_detected_time')
        
    #     # RA intruder collision avoidance
    #     # No RA intruder detection
    #     if ra_intruder == 0:
    #         ra_punishment_closeness = 0
    #     # RA intruder detected
    #     else:
    #         normed_nmac_distance = self.agent.nmac_radius / self.agent.detection_radius
    #         ra_punishment_closeness = -math.exp(normed_nmac_distance - dist_to_ra * 10.0)
        
    #     # Add proximity penalties to total reward
    #     reward += uav_punishment_closeness + ra_punishment_closeness
        
    #     return reward