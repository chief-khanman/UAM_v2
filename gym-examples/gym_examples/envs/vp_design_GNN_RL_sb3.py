# imports 
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
from map_env_revised import MapEnv


class VertiportGraphBuilder:
    """
    Builds and manages graph representations of vertiport configurations
    """
    # updated node feature dimension to 8 (includes selected vertiports as a feature)
    def __init__(self, airspace, node_feature_dim=8, edge_feature_dim=7, 
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
        """Create bidirectional mapping between vertiports and indices"""
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
        
        for idx, vertiport in self.idx_to_vertiport.items():
            features = []
            
            if vertiport in selected_vertiports:
                #                             need to ensure this fits with overall logic
                features.append(1.0) #F1    # this feature is for indicating vertiport is selected 
            else:
                features.append(0.0)
            # Location features (normalized)
            features.append(vertiport.location.x) #F2
            features.append(vertiport.location.y) #F3
            
            # Add metrics if available
            # metrics is a dict
            # metrics[key] -> dict; a dict within a dict 

            # metrics is a dict 
            # one of the KEYS of metrics is 'vertiport_metrics'
            # 'vertiport_metrics' holds dict[vertiport]: demand, ......
            if metrics and 'vertiport_metrics' in metrics:
                vp_metrics = metrics['vertiport_metrics'].get(vertiport, {})
                features.extend([
                    vp_metrics.get('demand', 1.0), #F4
                    vp_metrics.get('utilization', 1.0), #F5
                    vp_metrics.get('wait_time', 1.0), #F6
                    vp_metrics.get('throughput', 1.0), #F7
                ])
            else:
                # Default values if no metrics
                features.extend([1.0, 1.0, 1.0, 1.0])
            
            # Region encoding (one-hot or region id)
            region_id = self._get_region_id(vertiport)
            features.append(float(region_id)) #F8
            
            # Pad to node_feature_dim if needed
            while len(features) < self.node_feature_dim:
                features.append(0.0)
            
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
        features = []
        
        # Distance (primary feature for reward)
        distance = self.compute_distance(vp_i, vp_j) #f1
        features.append(distance) #f1
        
        # Edge type encoding
        if edge_type == 'inter':
            features.extend([1.0, 0.0, 0.0]) #f 2,3,4
        elif edge_type == 'intra':
            features.extend([0.0, 1.0, 0.0])
        else:  # 'full'
            features.extend([0.0, 0.0, 1.0])
        
        # Add flow metrics if available
        if metrics and 'flow_metrics' in metrics:
            flow = metrics['flow_metrics'].get((vp_i, vp_j), {})
            features.extend([
                flow.get('traffic', 1.0), #f5
                flow.get('capacity_used', 1.0), #f6
            ])
        else:
            features.extend([1.0, 1.0]) # f5, f6

        # IF vp_i and vp_j belong to the selected_vertiports list boolean feature is 1 else 0 
        if (vp_i in selected_vertiports) and (vp_j in selected_vertiports):
            features.append(1.0) #f7
        else:
            features.append(0.0) #f7
        
        # Pad to edge_feature_dim
        while len(features) < self.edge_feature_dim:
            features.append(0.0) #f8 ...
        
        return torch.tensor(features[:self.edge_feature_dim], dtype=torch.float32)
    
    def indices_to_vertiports(self, indices):
        """Convert tensor of indices to list of vertiports"""
        return [self.idx_to_vertiport[idx.item()] for idx in indices]
    
    def vertiports_to_indices(self, vertiports):
        """Convert list of vertiports to tensor of indices"""
        return torch.tensor([self.vertiport_to_idx[vp] for vp in vertiports])
    
    def region_idx_to_vertiport(self, action):
        #TODO: check if the implementation is correct 
        # example action: np.array([1,2,3,4])
        # region 1 vp id 1, region 2 vp id 2, region 3 vp id 3, region 4 vp id - previous vp id (since index of vertiports run from 0to3, 4 means no change of vp)
        new_vp_list = []
        for action_val, pack in zip(action, self.airspace.region_dict.items()):
            region, vp_list = pack[0], pack[1]
            if action_val < len(vp_list):
                _vp = vp_list[action_val]
                new_vp_list.append(_vp)
            else:
                _vp = self.current_vertiport_list[region]
        pass









# BUILD the vertiport design environment 
class VertiportDesignEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, simulator_step, total_episode_timestep):
        super().__init__()

        # define UAM simulator 
        test_mode = True
        num_regions = 4

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
            self.uam_simulator.airspace.make_regions_dict_vp_des_test_mode(map_centeroid_to_region_center=2*(32_000_000**0.5), region_center_to_vp=32_000_000**0.5)
        else:
            self.uam_simulator.airspace.make_regions_dict_vp_des('commercial', num_regions=num_regions)




        # define graph builder 
        #TODO: change node feature to have bool feature for active/inactive vertiport, location x,y(normalized), region_id_one_hot
        #TODO: change edge feature to have bool feature for active/inactive edge between vps, edge distance(normalized)
        self.graph_builder:VertiportGraphBuilder = VertiportGraphBuilder(
                                                                            airspace=self.uam_simulator.airspace,
                                                                            node_feature_dim=8, # x,y, 1,1,1,1,region_id
                                                                            edge_feature_dim=7,
                                                                            connectivity_type='full' # option1: inter_intra
                                                                        )
        
        
        
        
        
        # Define action and observation space
        # They must be gym.spaces objects
        num_vertiports = len(self.uam_simulator.airspace.vertiport_list)
        # Action space is discrete it changes depending on the region we are currently working with 
        # can the action space be a dictionary - region:vertiport_list + no_action
        self.action_space = spaces.MultiDiscrete([len(vp) + 1 for vp in self.uam_simulator.airspace.regions_dict.values()])
        
        # This is the node_feat, edge_attr, and edge_index derived from graph_builder
        self.observation_space = spaces.Dict({
            'node_feat':spaces.Box(low=-10000, high=10000, shape=(num_regions, num_vertiports)), 
            'edge_index':spaces.Box(low=0, high=100, shape=(2, num_vertiports)), 
            'edge_attr':spaces.Box(low=-10000, high=10000, shape=(num_vertiports,num_vertiports))
        })

        self.current_time_step = 0
        self.total_episode_timestep = total_episode_timestep
    
    
    def _collect_simulator_metrics(self, simulator:MapEnv):
        """Collect relevant metrics from simulator"""

        metrics = {
            'vertiport_metrics': {},
            'flow_metrics': {},
        }
        
        # Example metrics - customize based on your simulator
        try:
            metrics['total_throughput'] = getattr(simulator, 'total_throughput', 0.0)
            metrics['avg_wait_time'] = getattr(simulator, 'avg_wait_time', 0.0)
            metrics['avg_delay'] = getattr(simulator, 'avg_delay', 0.0)
        except:
            pass
        
        return metrics
    

    def compute_reward(self, selected_vertiports, simulator_metrics=None):
        """
        Compute reward based on selected vertiports
        
        For design problem, reward can be:
        1. Negative total distance (minimize distance)
        2. Simulator performance metrics
        3. Combination of both
        
        Args:
            selected_vertiports: List of selected vertiports
            simulator_metrics: Metrics from simulator (if available)
        
        Returns:
            reward: Scalar reward value
        """
        #TODO: create a internal function: _calculate_distance(), this function will take list of vertiports, and calculate the total_distance between all of them
        #      This _calculate_distance will be used on new_vp and old_vp then reward will be their difference 
        #      Same internal method for metrics, and then similar reward from difference of metrics. 
        total_distance = 0.0
        for i, vp_i in enumerate(selected_vertiports):
            for j, vp_j in enumerate(selected_vertiports):
                if i < j:  # Count each pair once
                    distance = self.graph_builder.compute_distance(vp_i, vp_j)
                    total_distance += distance
        
        #REWARD trick - 
        # best_vertiports = [(623157.,3355240.), (617501., 3355240.), (617501., 3349584.), (623157.,3349584.)]
        # best_count = 0
        # print('checking for best vp comb')
        # for vertiport in selected_vertiports:
        #     # print('inside check')
        #     # time.sleep(1)
        #     temp_xy = (vertiport.x, vertiport.y)
        #     # print(f'Temp vp tuple: {temp_xy}')
        #     if temp_xy in best_vertiports:
        #         best_count+=1
        #         # print(f'current best_count {best_count}')
        # if best_count == 4:
        #     distance_reward = 1000000
        #     time.sleep(3)
        #     print('Found best arrangement')
        #     return distance_reward, total_distance
        #     # print('FOUND BEST vp arrangement')
        #     # time.sleep(3)
        # else:
        #     # Reward is negative distance (we want to minimize total distance)
        #     distance_reward = -total_distance

        
        
        # Add simulator metrics if available
        if simulator_metrics:
            # Example: combine with throughput, wait time, etc.
            sim_reward = (
                simulator_metrics.get('total_throughput', 0.0) * 10.0 -
                simulator_metrics.get('avg_wait_time', 0.0) * 5.0 -
                simulator_metrics.get('avg_delay', 0.0) * 3.0
            )
            # Weighted combination
            reward = -total_distance # 0.3 * distance_reward + 0.7 * sim_reward
        else:
            reward = -total_distance
        
        return reward, total_distance  # Return both for tracking
    


    def _run_simulator(self,action):
        # One step of vp design problem means running the simulator and returning metrics 
        # action - new_vp_selection
        new_vertiport_list = self.graph_builder.region_idx_to_vertiport(action)
        self.current_selected_vertiports = new_vertiport_list
        self.uam_simulator.airspace.set_vertiport_list_vp_design(new_vertiport_list)
        
        #! ENV
        # Run simulator
        obs_simulator, info_simulator = self.uam_simulator.reset(seed=None)
            
        for sim_step in range(self.simulator_steps):
            obs, sim_reward, terminated, truncated, info = self.uam_simulator.step(
                self.uam_simulator.action_space.sample()
            )
            if terminated or truncated:
                break
            
        # Collect metrics from simulator
        #! NEW_STATE, s'
        metrics = self._collect_simulator_metrics(self.uam_simulator)
        
        return metrics

        #TODO: need action translation layer - will take list of discrete numbers and convert to vertiports/no_change 
        # BLUEPRINT: if action_value < len(), use action value to index into that vertiport from region_vertiport_list
        # BLUEPRINT: else, use no_action, meaning do not change current_vertiport for region
    
    def step(self, action):

        self.current_time_step += 1 
        

        # Collect metrics from simulator
        #! NEW_STATE, s'
        self.current_metrics = self._run_simulator(action)
        x, edge_index, edge_attr  = self.graph_builder.build_graph(self.current_selected_vertiports, self.current_metrics)

        observation = spaces.Dict({'node_feat':x,
                                   'edge_index':edge_index,
                                   'edge_attr':edge_attr})
        
        info = spaces.Dict({'node_feat':x,
                                   'edge_index':edge_index,
                                   'edge_attr':edge_attr})
        
        # Compute reward for this NEW configuration
        # reward - compute_reward
        #! REWARD 
        reward, total_distance = self.compute_reward(action, self.current_metrics) #! should the reward be the difference between previous state total distance and new state total distance 
        # Observation - metrics 
        
        # terminated/truncated are the same 
        terminated = False
        
        #! NEED TO ACCESS total_timestep from learn()
        if self.current_time_step > self.total_episode_timestep:
            truncated = True
        else:
            truncated = False
        


        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # define vertiport design variables
        # Track current selection AND metrics (STATE = config + metrics)
        self.current_selected_vertiports = None
        self.current_selected_indices = None
        self.current_metrics = None  # Store metrics as part of state!

        if self.current_selected_vertiports is None:
            self.current_selected_vertiports = []
            for region_id in sorted(self.graph_builder.airspace.regions_dict.keys()):
                first_vp = self.graph_builder.airspace.regions_dict[region_id][0]
                self.current_selected_vertiports.append(first_vp)
            self.current_selected_indices = self.graph_builder.vertiports_to_indices(
                self.current_selected_vertiports
            )
        


        # Use the selected vertiports to generate metric using the simulator
        self.current_metrics = self._run_simulator(self.current_selected_vertiports)

        # # build (alternative) simple_metrics 
        # # selected nodes 
        x, edge_index, edge_attr  = self.graph_builder.build_graph(self.current_selected_vertiports, self.current_metrics)
        # selected_nodes = np.array([])
        # # node location x,y (this might need to be normalized, use the center of map x,y as normalization)
        # node_location = np.array([(x,y),(x,y)])
        # # selected edges
        # selected_edges = np.array([])
        # # edge distance 
        # edge_distance = np.array([])
        # self.current_metrics = spaces.Dict({'selected_node':selected_nodes,
        #                              'node_location':node_location,
        #                              'selected_edges':selected_edges,
        #                              'edge_distance':edge_distance})
        # Observation - metrics - initial metrics built using first vertiports from each region 
        # Additional Observation - selected vertiports
        # use the other feats that are not used as additional_info
        observation = spaces.Dict({'node_feat':x, 'edge_index':edge_index, 'edge_attr':edge_attr})
        # info 
        info = {'node_feat':x, 'edge_index':edge_index, 'edge_attr':edge_attr}
        return observation, info

    def render(self):
        pass

    def close(self):
        pass



##### Env check #####
# comment out after checking 
from stable_baselines3.common.env_checker import check_env

env = VertiportDesignEnv(None)
# It will check your custom environment and output additional warnings if needed
check_env(env)

##### Env check #####

#build the observation space converter/wrapper 



class CustomGNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    #! how to convert observation to x, edge_index, and edge_attr for GNN input. 
    def __init__(self, node_features, edge_features, hidden_dim=128, num_layers=3, observation_space: spaces.Box, features_dim: int = 256):
        """
        Args:
            node_features (int): Number of input node features
            edge_features (int): Number of input edge features
            hidden_dim (int): Hidden dimension for embeddings
            num_layers (int): Number of GNN layers
        """
        super().__init__(observation_space, features_dim)

        
        
        self.node_proj = nn.Linear(node_features, hidden_dim)
        self.edge_proj = nn.Linear(edge_features, hidden_dim)

        self.conv_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, edge_dim=hidden_dim, heads=4, concat=False)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, node_features]
            edge_index (torch.Tensor): Edge connectivity [2, num_edges]
            edge_attr (torch.Tensor): Edge features [num_edges, edge_features]
            batch (torch.Tensor, optional): Batch assignment for nodes
        
        Returns:
            torch.Tensor: Node embeddings [num_nodes, hidden_dim]
        """
        # build the graph using observation
        #!                                            build_graph() args: selected_vp, metrics
        x, edge_index, edge_attr = VertiportDesignEnv.build_graph(observations)
        
        # Project inputs to hidden dimension
        h = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        # Apply GNN layers with residual connections
        for conv, norm in zip(self.conv_layers, self.layer_norms):
            h_new = conv(h, edge_index, edge_attr)
            h_new = F.relu(h_new)
            h = norm(h + h_new)  # Residual connection
            
        return h

policy_kwargs = dict(
    features_extractor_class=CustomGNN,
    features_extractor_kwargs=dict(features_dim=128),
)



##### Model training #####
# Instantiate the env
env = VertiportDesignEnv()

# Define and Train the agent
# PPO 
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(1000)

#A2C
model = A2C("MplPolicy", env).learn(total_timesteps=1000)

##### Model training #####










#run train the policy network 