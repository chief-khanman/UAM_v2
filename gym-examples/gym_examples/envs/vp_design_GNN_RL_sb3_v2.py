# imports 
import time
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from vertiport import Vertiport
from airspace import Airspace
from map_env_revised import MapEnv


class VertiportGraphBuilder:
    """
    Builds and manages graph representations of vertiport configurations
    """
    # updated node feature dimension to 8 (includes selected vertiports as a feature)
    def __init__(self, 
                 airspace:Airspace, 
                 node_feature_dim=3, 
                 edge_feature_dim=2, 
                 connectivity_type='inter_intra'):
        """
        Args:
            airspace: The airspace object containing regions and vertiports
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            connectivity_type: 'full' or 'inter_intra'
        """
        self.airspace = airspace
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.connectivity_type = connectivity_type
        
        # Build vertiport to index mapping
        self.vertiport_to_idx = {}
        self.idx_to_vertiport = {}
        self._build_vertiport_mapping()
        
        # Build region mask
        self.region_mask = self._build_region_mask()
        
    def _build_vertiport_mapping(self):
        """
        Builds vertiport to index. 
        Create bidirectional mapping between vertiports and indices
        """

        idx = 0
        for region_id, vertiport_list in self.airspace.regions_dict.items():
            for vertiport in vertiport_list:
                self.vertiport_to_idx[vertiport] = idx
                self.idx_to_vertiport[idx] = vertiport
                idx += 1
    
    def _build_region_mask(self):
        """
        Create region mask [num_regions, (total)num_vertiports]
        mask[i, j] = 1 if vertiport j belongs to region i
        """
        num_regions = len(self.airspace.regions_dict)
        num_vertiports = len(self.vertiport_to_idx)
        
        mask = torch.zeros(num_regions, num_vertiports, dtype=torch.float32)
        
        for region_id, vertiport_list in self.airspace.regions_dict.items():
            for vertiport in vertiport_list:
                vp_idx = self.vertiport_to_idx[vertiport]
                mask[region_id, vp_idx] = 1.0
        
        return mask
    
    def compute_distance(self, vp1:Vertiport, vp2:Vertiport):
        """Compute Euclidean distance between two vertiports"""
        return vp1.location.distance(vp2.location)
    
    def build_graph(self, selected_vertiports=None, metrics=None):
        """
        Build graph with node features, edge index, and edge attributes
        
        Args:
            selected_vertiports: List of currently selected vertiports (one per region)
            metrics: Dictionary containing simulator metrics for features
        
        Returns:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feature_dim]
        """
        #! do we need this variable ??
        num_vertiports = len(self.vertiport_to_idx)
        
        # Build node features
        x = self._build_node_features(selected_vertiports, metrics)
        
        # Build edge connectivity and features
        if self.connectivity_type == 'full':
            edge_index, edge_attr = self._build_fully_connected_graph(metrics, selected_vertiports)
        else:  # 'inter_intra'
            edge_index, edge_attr = self._build_inter_intra_graph(
                selected_vertiports, metrics
            )
        
        return x, edge_index, edge_attr
    #TODO: rework this method to only use selected_vp, normalized_location as features - remove other features
    def _build_node_features(self, selected_vertiports, metrics=None):
        """
        Build node features for each vertiport
        
        Features can include:
        - Selected vertiports
        - Location (x, y normalized)
        - Capacity
        - Current demand/utilization
        - Historical metrics
        - Region encoding
        """
        num_vertiports = len(self.vertiport_to_idx)
        x = torch.zeros(num_vertiports, self.node_feature_dim)
        
        # normalization calculation
        #! alternative normalization technique - 
        #  minx, miny, maxx, maxy = self.airspace.location_utm_gdf.total_bounds

        all_x = [vp.location.x for vp in self.idx_to_vertiport.values()]
        all_y = [vp.location.y for vp in self.idx_to_vertiport.values()]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        # Avoid division by zero
        range_x = max_x - min_x if max_x != min_x else 1.0
        range_y = max_y - min_y if max_y != min_y else 1.0

        for idx, vertiport in self.idx_to_vertiport.items():
            features = []
            
            if vertiport in selected_vertiports:
                #                             need to ensure this fits with overall logic
                features.append(1.0) #F1    # this feature is for indicating vertiport is selected 
            else:
                features.append(0.0)

            # F2, F3: Normalized location features (0 to 1 range)
            normalized_x = (vertiport.location.x - min_x) / range_x
            normalized_y = (vertiport.location.y - min_y) / range_y
            features.append(normalized_x)
            features.append(normalized_y)
            
            # remove this line - once the problem is ready for deployment 
            x[idx] = torch.tensor(features[:self.node_feature_dim])
        
        return x
    
    def _get_region_id(self, vertiport):
        """Get region ID for a vertiport"""
        for region_id, vp_list in self.airspace.regions_dict.items():
            if vertiport in vp_list:
                return region_id
        return -1
    
    #TODO: rework this method to only use selected_vp, normalized_location as features - remove other features
    def _build_fully_connected_graph(self, metrics=None, selected_vertiports=None):
        """Build fully connected graph"""
        num_vertiports = len(self.vertiport_to_idx)
        edges = []
        edge_attrs = []
        
        for i in range(num_vertiports):
            for j in range(num_vertiports):
                if i != j:
                    edges.append([i, j])
                    vp_i = self.idx_to_vertiport[i]
                    vp_j = self.idx_to_vertiport[j]
                    edge_attr = self._compute_edge_features(vp_i, vp_j, 'full', metrics, selected_vertiports)
                    edge_attrs.append(edge_attr)
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_attr = torch.stack(edge_attrs)
        
        return edge_index, edge_attr
    
    def _build_inter_intra_graph(self, selected_vertiports, metrics=None):
        """
        Build graph with inter-region and intra-region edges
        
        - Selected vertiports: connected to all other selected vertiports (inter-region)
        - All vertiports in a region: connected to each other (intra-region)
        """
        edges = []
        edge_attrs = []
        
        if selected_vertiports is None:
            # Default: select first vertiport from each region
            selected_vertiports = []
            for region_id in sorted(self.airspace.regions_dict.keys()):
                selected_vertiports.append(self.airspace.regions_dict[region_id][0])
        
        # Convert vertiports to indices
        selected_indices = [self.vertiport_to_idx[vp] for vp in selected_vertiports]
        
        # Inter-region edges (between selected vertiports)
        for i, idx_i in enumerate(selected_indices):
            for j, idx_j in enumerate(selected_indices):
                if i != j:
                    edges.append([idx_i, idx_j])
                    vp_i = self.idx_to_vertiport[idx_i]
                    vp_j = self.idx_to_vertiport[idx_j]
                    edge_attr = self._compute_edge_features(vp_i, vp_j, 'inter', metrics)
                    edge_attrs.append(edge_attr)
        
        # Intra-region edges (within each region)
        for region_id, vp_list in self.airspace.regions_dict.items():
            region_indices = [self.vertiport_to_idx[vp] for vp in vp_list]
            
            for i, idx_i in enumerate(region_indices):
                for j, idx_j in enumerate(region_indices):
                    if i != j:
                        edges.append([idx_i, idx_j])
                        vp_i = self.idx_to_vertiport[idx_i]
                        vp_j = self.idx_to_vertiport[idx_j]
                        edge_attr = self._compute_edge_features(vp_i, vp_j, 'intra', metrics)
                        edge_attrs.append(edge_attr)
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_attr = torch.stack(edge_attrs)
        
        return edge_index, edge_attr
    #TODO: rework this method to only use selected_vp, normalized_location as features - remove other features
    def _compute_edge_features(self, vp_i, vp_j, edge_type, metrics=None, selected_vertiports=None):
        """
        Compute edge features between two vertiports
        
        Features include:
        - Distance
        - Edge type (inter/intra/full)
        - Flow metrics if available
        - For a selected pair of vertiports activate boolean edge feature
        """
        
        all_x = [vp.location.x for vp in self.idx_to_vertiport.values()]
        all_y = [vp.location.y for vp in self.idx_to_vertiport.values()]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        diagonal = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
        
        features = []
        
        # Distance (primary feature for reward) - normalize distance
        distance = self.compute_distance(vp_i, vp_j) #f1 
        features.append(distance/diagonal) #f1 - normalized by diagonal
        
        # Boolean feature indicating if both endpoints are selected
        if selected_vertiports is not None and (vp_i in selected_vertiports) and (vp_j in selected_vertiports):
            features.append(1.0) #f2
        else:
            features.append(0.0) #f2
        
        # Edge type encoding - only full connection will be used - remove this feature 
        if edge_type == 'inter':
            features.extend([1.0, 0.0, 0.0]) #f 3,4,5
        elif edge_type == 'intra':
            features.extend([0.0, 1.0, 0.0])
        else:  # 'full'
            features.extend([0.0, 0.0, 1.0])
        
        # Add flow metrics if available
        if metrics and 'flow_metrics' in metrics:
            flow = metrics['flow_metrics'].get((vp_i, vp_j), {})
            features.extend([
                flow.get('traffic', 1.0), #f6
                flow.get('capacity_used', 1.0), #f7
            ])
        else:
            features.extend([1.0, 1.0]) # f6, f7
        
        # Pad to edge_feature_dim - no padding required 
        while len(features) < self.edge_feature_dim:
            features.append(0.0)
        
        return torch.tensor(features[:self.edge_feature_dim], dtype=torch.float32)
    
    def indices_to_vertiports(self, indices):
        """Convert tensor of indices to list of vertiports"""
        return [self.idx_to_vertiport[idx.item()] for idx in indices]
    
    def vertiports_to_indices(self, vertiports):
        """Convert list of vertiports to tensor of indices"""
        return torch.tensor([self.vertiport_to_idx[vp] for vp in vertiports])
    
    def region_idx_to_vertiport(self, action):
        """
        Use action array to build new vertiport list.
        
        Args:
            action: np.array where each element is either:
                - 0 to len(vp_list): select that vertiport from the region
                
        
        Returns:
            new_vp_list: list of vertiports after applying action
        """

        new_vp_list = []
        for region, vp_list in self.airspace.regions_dict.items():
            # vp_list is sorted and unchanging list, 
            # so action is used to access the index location of vp_list to access a vertiport
            # this gives consistency of vertiport and lets the policy learn about vertiport's impact on space desing 
            _vp = vp_list[action[region]] 
            new_vp_list.append(_vp)
        
        return new_vp_list


# BUILD the vertiport design environment 
class VertiportDesignEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, simulator_step, total_episode_timestep, node_feat_dim=3, edge_feat_dim=2):
        super().__init__()

        # define UAM simulator 
        test_mode = True
        self.num_regions = 4

        TEST_MODE = True
    
        if TEST_MODE:
            vertiport_tag_list = []
        else:
            vertiport_tag_list = [('building', 'commercial')]
        
        self.simulator_step = simulator_step
        self.uam_simulator = MapEnv(
                                number_of_uav=0,
                                num_ORCA_uav=0,
                                number_of_vertiport=100,
                                location_name='Austin, Texas, USA',
                                airspace_tag_list=[],
                                vertiport_tag_list=vertiport_tag_list,
                                max_episode_steps=100,
                                number_of_other_agents_observed_for_model=7,
                                sleep_time=0,
                                seed=70,
                                obs_space_str='UAM_UAV',
                                sorting_criteria='closest_first',
                                render_mode=None,
                                max_uavs=100,
                                max_vertiports=150,
                                make_uav_at_timestep=300,
                                vp_design_problem=True
                            )
        self.uam_simulator.set_airspace_vp_design()

        if test_mode:
            print('Vertiport Design problem in test mode')
            
            #self.uam_simulator.airspace.make_regions_dict_vp_des_test_mode(map_centeroid_to_region_center=2*(32_000_000**0.5), region_center_to_vp=32_000_000**0.5)
            
            self.uam_simulator.airspace.make_regions_dict_vp_des_test_mode_new(number_of_regions=4, 
                                                number_of_vertiports=5,
                                                spacing_between_vertiports=15000,
                                                orientation='horizontal',
                                                distance_center_2_vertex=5000,
                                                distance_map_centeroid_2_region_center=15000,
                                                )
        else:
            self.uam_simulator.airspace.make_regions_dict_vp_des('commercial', num_regions=self.num_regions)

        # define graph builder 
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.graph_builder:VertiportGraphBuilder = VertiportGraphBuilder(
                                                                            airspace=self.uam_simulator.airspace,
                                                                            node_feature_dim=self.node_feat_dim,
                                                                            edge_feature_dim=self.edge_feat_dim,
                                                                            connectivity_type='full'
                                                                        )
        
        # Define action and observation space
        num_vertiports = len(self.uam_simulator.airspace.vertiport_list)
        
        # Action space: for each region, select vertiport index (0 to len(vp_list))
        self.action_space = spaces.MultiDiscrete([len(vp_list) for vp_list in self.uam_simulator.airspace.regions_dict.values()])
        
        max_edges = num_vertiports * (num_vertiports - 1)
        
        # Observation space
        self.observation_space = spaces.Dict({
            'node_feat': spaces.Box(low=-10000, 
                                    high=10000, 
                                    shape=(num_vertiports, self.graph_builder.node_feature_dim),
                                    dtype=np.float32), 
            'edge_index': spaces.Box(low=-10, 
                                     high=100, 
                                     shape=(2, int(max_edges)),
                                     dtype=np.int64), 
            'edge_attr': spaces.Box(low=-1000000., 
                                    high=1000000., 
                                    shape=(int(max_edges), self.graph_builder.edge_feature_dim),
                                    dtype=np.float32),
            'selected_actions': spaces.Box(low=0,
                                           high=num_vertiports,
                                           shape=(self.num_regions,),
                                           dtype=np.int64)
        })

        self.current_time_step = 0
        self.total_episode_timestep = total_episode_timestep
        
        # State tracking variables (initialized in reset)
        self.current_selected_vertiports = None
        self.previous_selected_vertiports = None
        self.current_total_distance = None
        self.previous_total_distance = None
        self.current_metrics = None
        self.previous_metrics = None
    
    def _calculate_total_distance(self, selected_vertiports):
        """
        Calculate total pairwise distance between all selected vertiports.
        
        Args:
            selected_vertiports: List of selected vertiports (one per region)
        
        Returns:
            total_distance: Sum of distances between all pairs
        """
        total_distance = 0.0
        for i, vp_i in enumerate(selected_vertiports):
            for j, vp_j in enumerate(selected_vertiports):
                if i < j:  # Count each pair once
                    distance = self.graph_builder.compute_distance(vp_i, vp_j)
                    total_distance += distance
        return total_distance
    
    def _collect_simulator_metrics(self, simulator: MapEnv):
        """Collect relevant metrics from simulator"""
        metrics = {
            'vertiport_metrics': {},
            'flow_metrics': {},
        }
        
        try:
            metrics['total_throughput'] = getattr(simulator, 'total_throughput', 0.0)
            metrics['avg_wait_time'] = getattr(simulator, 'avg_wait_time', 0.0)
            metrics['avg_delay'] = getattr(simulator, 'avg_delay', 0.0)
        except:
            pass
        
        return metrics

    def _compute_reward(self):
        """
        Compute dense improvement reward based on distance reduction.
        
        Reward = previous_total_distance - current_total_distance
        
        Positive reward when distance decreases (improvement).
        Negative reward when distance increases (worse configuration).
        
        Returns:
            reward: Scalar improvement reward
        """
        if self.previous_total_distance is None:
            # First step after reset - no previous distance to compare
            # Return 0 reward for the first step
            return 0.0
        
        # Improvement reward: positive when distance decreases
        reward = self.previous_total_distance - self.current_total_distance
        
        return reward

    def _run_simulator(self, action):
        """
        Run one step of the vertiport design problem.
        Updates current_selected_vertiports based on action and runs simulator.
        
        Args:
            action: Array of action values for each region
        
        Returns:
            metrics: Dictionary of simulator metrics
        """
        # Convert action to new vertiport list
        new_vertiport_list = self.graph_builder.region_idx_to_vertiport(action)
        
        self.current_selected_vertiports = new_vertiport_list
        
        # Update simulator with new vertiport configuration
        self.uam_simulator.airspace.set_vertiport_list_vp_design(new_vertiport_list)
        
        # Run simulator
        obs_simulator, info_simulator = self.uam_simulator.reset(seed=None)
            
        for sim_step in range(self.simulator_step):
            obs, sim_reward, terminated, truncated, info = self.uam_simulator.step(
                self.uam_simulator.action_space.sample()
            )
            if terminated or truncated:
                break
            
        # Collect metrics from simulator
        metrics = self._collect_simulator_metrics(self.uam_simulator)
        
        return metrics
    
    def step(self, action):
        """
        Execute one step of the environment.
        
        Args:
            action: Array where each element selects vertiport for that region
                   (value < len(vp_list) selects that vp, value == len(vp_list) keeps current)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_time_step += 1 
        
        # Save previous state for reward computation
        self.previous_selected_vertiports = self.current_selected_vertiports.copy()
        self.previous_total_distance = self.current_total_distance
        self.previous_metrics = self.current_metrics

        # Run simulator with new action (updates current_selected_vertiports)
        self.current_metrics = self._run_simulator(action)
        
        # Compute new total distance after action
        self.current_total_distance = self._calculate_total_distance(self.current_selected_vertiports)
        
        # Build graph observation with new configuration
        x, edge_index, edge_attr = self.graph_builder.build_graph(
            self.current_selected_vertiports, self.current_metrics
        )
        observation = {
            'node_feat': x.numpy().astype(np.float32),
            'edge_index': edge_index.numpy().astype(np.int64),
            'edge_attr': edge_attr.numpy().astype(np.float32),
            'selected_actions': action
        }

        # Compute dense improvement reward
        reward = self._compute_reward()
        
        # Episode termination conditions
        terminated = False
        truncated = self.current_time_step >= self.total_episode_timestep
        
        # Info for debugging/logging
        info = {
            'node_feat': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr, 
            'selected_vertiports': self.current_selected_vertiports,
            'previous_total_distance': self.previous_total_distance,
            'current_total_distance': self.current_total_distance,
            'reward': reward,
            'timestep': self.current_time_step
        }

        if terminated or truncated: 
            print('Best vertiports are: (631282.60 3355349) OR (631282.60 3349471) \n (621872 3362655) \n(610327 3352410) \n(620327 3337410)')
            print('\n')
            print(f'Final selected vertiports are: {self.current_selected_vertiports}')
            #time.sleep(0.2)
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Returns:
            observation, info
        """
        super().reset(seed=seed)
        
        # Reset timestep counter
        self.current_time_step = 0

        # Initialize with first vertiport from each region
        self.current_selected_vertiports = []
        for region_id in sorted(self.graph_builder.airspace.regions_dict.keys()):
            first_vp = self.graph_builder.airspace.regions_dict[region_id][0]
            self.current_selected_vertiports.append(first_vp)

        self.current_selected_indices = self.graph_builder.vertiports_to_indices(
            self.current_selected_vertiports
        )
        
        # Initialize previous state trackers to None (first step has no previous)
        self.previous_selected_vertiports = None
        self.previous_total_distance = None
        self.previous_metrics = None

        # Create initial action that selects first VP in each region
        num_regions = len(self.uam_simulator.airspace.regions_dict)
        initial_action = np.array([0] * num_regions)

        # Run simulator to get initial metrics
        self.current_metrics = self._run_simulator(initial_action)
        
        # Compute initial total distance
        self.current_total_distance = self._calculate_total_distance(self.current_selected_vertiports)
        
        # Build initial graph observation
        x, edge_index, edge_attr = self.graph_builder.build_graph(
            self.current_selected_vertiports, self.current_metrics
        )
        observation = {
            'node_feat': x.numpy().astype(np.float32),
            'edge_index': edge_index.numpy().astype(np.int64),
            'edge_attr': edge_attr.numpy().astype(np.float32),
            'selected_actions': initial_action
        }
        
        info = {
            'node_feat': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr, 
            'selected_vertiports': self.current_selected_vertiports,
            'initial_total_distance': self.current_total_distance,
            'reward': 0.0,
            'timestep': self.current_time_step
        }
        
        return observation, info

    def render(self):
        pass

    def close(self):
        pass

    def _get_selected_actions(self):
        """
        Get the global vertiport indices of currently selected vertiports.
        
        Returns:
            np.array of vertiport indices (one per region)
        """
        selected_actions = []
        for vp in self.current_selected_vertiports:
            selected_action = self.graph_builder.vertiport_to_idx[vp]
            selected_actions.append(selected_action)
        return np.array(selected_actions)





class CustomGNN(BaseFeaturesExtractor):
    """
    Custom GNN feature extractor for graph-structured observations.
    
    Processes graph observations (node_feat, edge_index, edge_attr) through
    GAT layers and concatenates selected_actions to produce final features.
    
    :param observation_space: (gym.spaces.Dict) Observation space containing:
        - node_feat: [num_nodes, node_feat_dim]
        - edge_index: [2, num_edges]
        - edge_attr: [num_edges, edge_feat_dim]
        - selected_actions: [num_regions]
    :param features_dim: (int) Hidden dimension for GNN layers
    :param final_dim: (int) Output dimension of GNN before concatenating selected_actions
    :param num_layers: (int) Number of GAT layers
    """
    
    def __init__(self, 
                 observation_space: spaces.Dict, 
                 features_dim: int = 16, 
                 final_dim: int = 8, 
                 num_layers: int = 1):
        
        # Calculate total output dimension (GNN output + selected_actions)
        num_regions = observation_space['selected_actions'].shape[0]
        total_output_dim = final_dim + num_regions
        
        # Must call super().__init__() with actual output dimension
        # This sets self._features_dim which policy network uses
        super().__init__(observation_space, total_output_dim)
        
        # Get input dimensions from observation space
        self.node_feat_dim = observation_space['node_feat'].shape[1]
        self.edge_feat_dim = observation_space['edge_attr'].shape[1]  # Fixed: edge_attr not edge_feat
        
        # Input projection layers
        self.node_proj = nn.Linear(self.node_feat_dim, features_dim)
        self.edge_proj = nn.Linear(self.edge_feat_dim, features_dim)

        # GAT convolution layers
        self.conv_layers = nn.ModuleList([
            GATConv(
                in_channels=features_dim, 
                out_channels=features_dim, 
                edge_dim=features_dim, 
                heads=4, 
                concat=False  # Average heads, output stays [num_nodes, features_dim]
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization for residual connections
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(features_dim) for _ in range(num_layers)
        ])

        # Output projection: features_dim -> final_dim
        self.output_proj = nn.Linear(features_dim, final_dim)

    def forward(self, observations) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        SB3 passes batched observations even with single env.
        Shapes:
            node_feat: [batch_size, num_nodes, node_feat_dim]
            edge_index: [batch_size, 2, num_edges]
            edge_attr: [batch_size, num_edges, edge_feat_dim]
            selected_actions: [batch_size, num_regions]
        
        Returns:
            torch.Tensor: [batch_size, final_dim + num_regions]
        """
        node_feat = observations['node_feat']
        edge_index = observations['edge_index'].long()
        edge_attr = observations['edge_attr']
        selected_actions = observations['selected_actions']
        
        batch_size = node_feat.shape[0]
        
        graph_embeddings = []
        
        # Process each graph in the batch
        for i in range(batch_size):
            # Extract single graph from batch
            x = node_feat[i]      # [num_nodes, node_feat_dim]
            ei = edge_index[i]    # [2, num_edges]
            ea = edge_attr[i]     # [num_edges, edge_feat_dim]
            
            # Project inputs to hidden dimension
            h = self.node_proj(x)         # [num_nodes, features_dim]
            ea_proj = self.edge_proj(ea)  # [num_edges, features_dim]
            
            # Apply GNN layers with residual connections
            for conv, norm in zip(self.conv_layers, self.layer_norms):
                h_new = conv(h, ei, ea_proj)
                h_new = F.relu(h_new)
                h = norm(h + h_new)  # Residual connection
            
            # Global mean pooling to get graph-level embedding
            graph_emb = torch.mean(h, dim=0)  # [features_dim]
            
            # Project to final dimension
            graph_emb = self.output_proj(graph_emb)  # [final_dim]
            
            graph_embeddings.append(graph_emb)
        
        # Stack graph embeddings: [batch_size, final_dim]
        graph_embeddings = torch.stack(graph_embeddings, dim=0)
        
        # Concatenate selected_actions: [batch_size, final_dim + num_regions]
        output = torch.cat([graph_embeddings, selected_actions.float()], dim=1)
        
        return output

if __name__ == '__main__':
    # Environment check
    from stable_baselines3.common.env_checker import check_env

    env = VertiportDesignEnv(simulator_step=3, total_episode_timestep=1000, node_feat_dim=3, edge_feat_dim=2)
    # It will check your custom environment and output additional warnings if needed
    # check_env(env)
    
    # print("Environment check passed!")
    
    # # Quick test of reward function
    # obs, info = env.reset()
    # print(f"Initial total distance: {env.current_total_distance}")
    
    # for i in range(5):
    #     action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     print(f"Step {i+1}: reward={reward:.2f}, prev_dist={info['previous_total_distance']:.2f}, curr_dist={info['current_total_distance']:.2f}")



    policy_kwargs = dict(
        features_extractor_class=CustomGNN,
        features_extractor_kwargs=dict(
            features_dim=16,   # Hidden dimension for GNN
            final_dim=8,       # GNN output dimension before concat
            num_layers=2       # Number of GAT layers
        ),
    )

    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    