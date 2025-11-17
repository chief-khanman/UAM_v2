import time
import torch
import torch.nn.functional as F
import numpy as np
from shapely.geometry import Point
from GNN_RL import StationSelectionGNNRL
from map_env_revised import MapEnv
from vertiport import Vertiport

class VertiportGraphBuilder:
    """
    Builds and manages graph representations of vertiport configurations
    """
    def __init__(self, airspace, node_feature_dim=20, edge_feature_dim=12, 
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
        Create region mask [num_regions, num_vertiports]
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
        num_vertiports = len(self.vertiport_to_idx)
        
        # Build node features
        x = self._build_node_features(metrics)
        
        # Build edge connectivity and features
        if self.connectivity_type == 'full':
            edge_index, edge_attr = self._build_fully_connected_graph(metrics)
        else:  # 'inter_intra'
            edge_index, edge_attr = self._build_inter_intra_graph(
                selected_vertiports, metrics
            )
        
        return x, edge_index, edge_attr
    
    def _build_node_features(self, metrics=None):
        """
        Build node features for each vertiport
        
        Features can include:
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
            
            # Location features (normalized)
            features.append(vertiport.location.x)
            features.append(vertiport.location.y)
            
            # Add metrics if available
            # metrics is a dict
            # metrics[key] -> dict; a dict within a dict 

            # metrics is a dict 
            # one of the KEYS of metrics is 'vertiport_metrics'
            # 'vertiport_metrics' holds dict[vertiport]: demand, ......
            if metrics and 'vertiport_metrics' in metrics:
                vp_metrics = metrics['vertiport_metrics'].get(vertiport, {})
                features.extend([
                    vp_metrics.get('demand', 0.0),
                    vp_metrics.get('utilization', 0.0),
                    vp_metrics.get('wait_time', 0.0),
                    vp_metrics.get('throughput', 0.0),
                ])
            else:
                # Default values if no metrics
                #TODO: change to some constant value instead of 0.0
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            # Region encoding (one-hot or region id)
            region_id = self._get_region_id(vertiport)
            features.append(float(region_id))
            
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
    
    
    def _build_fully_connected_graph(self, metrics=None):
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
                    edge_attr = self._compute_edge_features(vp_i, vp_j, 'full', metrics)
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
    
    def _compute_edge_features(self, vp_i, vp_j, edge_type, metrics=None):
        """
        Compute edge features between two vertiports
        
        Features include:
        - Distance
        - Edge type (inter/intra/full)
        - Flow metrics if available
        """
        features = []
        
        # Distance (primary feature for reward)
        distance = self.compute_distance(vp_i, vp_j)
        features.append(distance)
        
        # Edge type encoding
        if edge_type == 'inter':
            features.extend([1.0, 0.0, 0.0])
        elif edge_type == 'intra':
            features.extend([0.0, 1.0, 0.0])
        else:  # 'full'
            features.extend([0.0, 0.0, 1.0])
        
        # Add flow metrics if available
        if metrics and 'flow_metrics' in metrics:
            flow = metrics['flow_metrics'].get((vp_i, vp_j), {})
            features.extend([
                flow.get('traffic', 0.0),
                flow.get('capacity_used', 0.0),
            ])
        else:
            #TODO: change this to use [1,1]
            features.extend([0.0, 0.0])
        
        # Pad to edge_feature_dim
        while len(features) < self.edge_feature_dim:
            features.append(0.0)
        
        return torch.tensor(features[:self.edge_feature_dim], dtype=torch.float32)
    
    def indices_to_vertiports(self, indices):
        """Convert tensor of indices to list of vertiports"""
        return [self.idx_to_vertiport[idx.item()] for idx in indices]
    
    def vertiports_to_indices(self, vertiports):
        """Convert list of vertiports to tensor of indices"""
        return torch.tensor([self.vertiport_to_idx[vp] for vp in vertiports])


class A2CTrainer:
    """
    Advantage Actor-Critic trainer for vertiport selection (combinatorial optimization)
    """
    def __init__(self, model, graph_builder, lr=3e-4, gamma=0.99, 
                 value_coef=0.5, entropy_coef=0.01):
        self.model = model
        self.graph_builder:VertiportGraphBuilder = graph_builder
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Track current selection AND metrics (STATE = config + metrics)
        self.current_selected_vertiports = None
        self.current_selected_indices = None
        self.current_metrics = None  # Store metrics as part of state!
        
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
        # Distance-based component (negative because we want to minimize)
        # for vp in selected_vertiports:
        #     print(f'Vertiport x: {vp.x}, x type: {type(vp.x)}')
        #     time.sleep(2)


        total_distance = 0.0
        for i, vp_i in enumerate(selected_vertiports):
            for j, vp_j in enumerate(selected_vertiports):
                if i < j:  # Count each pair once
                    distance = self.graph_builder.compute_distance(vp_i, vp_j)
                    total_distance += distance
        
        #REWARD trick - 
        best_vertiports = [(623157.,3355240.), (617501., 3355240.), (617501., 3349584.), (623157.,3349584.)]
        best_count = 0
        print('checking for best vp comb')
        for vertiport in selected_vertiports:
            print('inside check')
            time.sleep(1)
            temp_xy = (vertiport.x, vertiport.y)
            print(f'Temp vp tuple: {temp_xy}')
            if temp_xy in best_vertiports:
                best_count+=1
                print(f'current best_count {best_count}')
        if best_count == 4:
            distance_reward = 10000
            print('FOUND BEST vp arrangement')
            time.sleep(3)
        else:
            # Reward is negative distance (we want to minimize total distance)
            distance_reward = -total_distance

        
        
        # Add simulator metrics if available
        if simulator_metrics:
            # Example: combine with throughput, wait time, etc.
            sim_reward = (
                simulator_metrics.get('total_throughput', 0.0) * 10.0 -
                simulator_metrics.get('avg_wait_time', 0.0) * 5.0 -
                simulator_metrics.get('avg_delay', 0.0) * 3.0
            )
            # Weighted combination
            reward = 0.3 * distance_reward + 0.7 * sim_reward
        else:
            reward = distance_reward
        
        return reward
    
    def train_step(self, graph_data, region_mask, reward, current_selection):
        """
        Single training step
        
        Args:
            graph_data: (x, edge_index, edge_attr) for the graph
            region_mask: Region assignments
            reward: Reward for current configuration
            current_selection: Currently selected vertiport indices
        """
        x, edge_index, edge_attr = graph_data
        
        # Forward pass
        #! why is action_change(bool) not used 
        #TODO: action_changed can be used as a signal to indicate for a given step of RL - we have selected a new vertiport for this simulation run 
        selected_stations, log_probs, value, entropy, action_changed = self.model(
            x, edge_index, edge_attr, region_mask, 
            current_selection=current_selection,
            training=True
        )
        
        # Compute advantage
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        # TD error
        #TODO: check if value is calculated using 1)current_vertiports and 2)new_vertiports/indicies
        #TODO: check the definition of advantage - make sure this is correct implementation -> R + V(s+1) - V(s)
        advantage = reward_tensor - value.detach()
        
        # Policy loss (Actor)
        #TODO: plot policy loss
        policy_loss = -(log_probs * advantage)
        
        # Value loss (Critic)
        #TODO: plot value loss
        value_loss = F.mse_loss(value, reward_tensor)
        
        # Entropy bonus (for exploration)
        entropy_loss = -entropy
        
        # Combined loss
        loss = (policy_loss + 
                self.value_coef * value_loss + 
                self.entropy_coef * entropy_loss)
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'value_estimate': value.item(),
            'reward': reward,
            'advantage': advantage.item(),
            'selected_stations': selected_stations.cpu().numpy()
        }
    
    def train_episode(self, simulator:MapEnv, simulator_steps, num_design_steps=1):
        """
        Train for one episode
        
        Flow:
        1. Start with previous vertiport selection AND metrics (STATE)
        2. GNN-RL proposes new selection (ACTION) based on current state
        3. Run simulator with new selection to get new metrics
        4. Compute reward for new configuration
        5. Train model on transition: (old_state, action, reward, new_state)
        6. Update current state (selection + metrics)
        
        Args:
            simulator: MapEnv instance
            simulator_steps: Number of steps to run simulator
            num_design_steps: Number of times to run design optimization in episode
        """
        episode_rewards = []
        episode_stats = []
        
        # Initialize with first vertiport from each region if no current selection
        if self.current_selected_vertiports is None:
            self.current_selected_vertiports = []
            for region_id in sorted(self.graph_builder.airspace.regions_dict.keys()):
                first_vp = self.graph_builder.airspace.regions_dict[region_id][0]
                self.current_selected_vertiports.append(first_vp)
            self.current_selected_indices = self.graph_builder.vertiports_to_indices(
                self.current_selected_vertiports
            )
            # Current metrics is None only for FIRST episode ever
            # After first episode, we'll have metrics from simulator
        # step(s) in episode  
        for design_step in range(num_design_steps):
            # Build graph with current selection AND PREVIOUS METRICS
            # This represents the TRUE state: (configuration, observed metrics)
            x, edge_index, edge_attr = self.graph_builder.build_graph(
                selected_vertiports=self.current_selected_vertiports,
                metrics=self.current_metrics  #FIX: Use stored metrics!
            )
            
            # Model proposes new selection (or keeps current)
            with torch.no_grad():
                new_selected_indices, _, _, _, action_changed = self.model(
                    x, edge_index, edge_attr, 
                    self.graph_builder.region_mask,
                )
            
            # Convert to vertiports
            new_selected_vertiports = self.graph_builder.indices_to_vertiports(
                new_selected_indices
            )
            
            # Set vertiports in simulator
            simulator.airspace.set_vertiport_list_vp_design(new_selected_vertiports)
            
            # Run simulator
            obs, info = simulator.reset(seed=None)
            
            for sim_step in range(simulator_steps):
                obs, sim_reward, terminated, truncated, info = simulator.step(
                    simulator.action_space.sample()
                )
                if terminated or truncated:
                    break
            
            # Collect metrics from simulator
            #! check this method and make sure this is using the initial_metric and end_metrics from map_env
            simulator_metrics = self._collect_simulator_metrics(simulator)
            
            # Compute reward for this NEW configuration
            reward = self.compute_reward(new_selected_vertiports, simulator_metrics)
            # CHECKING reward to make sure it makes sense 
            # print()
            # print(f'Vertiports: {new_selected_vertiports}')
            # print()
            # print(f'Reward from episode{episode} --- step{design_step}: {reward}')
            # print()
            # time.sleep(2)
            # DELETE after checking 
            episode_rewards.append(reward)
            
            # Rebuild graph with NEW selection and NEW metrics
            # This is the state the model will see NEXT time
            x_new, edge_index_new, edge_attr_new = self.graph_builder.build_graph(
                selected_vertiports=new_selected_vertiports,
                metrics=simulator_metrics
            )
            
            # Training step: learn from transition
            # Old state: (x, edge_index, edge_attr) with old metrics
            # Action: selected new vertiports
            # Reward: from simulator
            # New state: (x_new, ...) with new metrics (not used in A2C, but available)
            stats = self.train_step(
                (x_new, edge_index_new, edge_attr_new),  # Use new state with metrics
                self.graph_builder.region_mask,
                reward,
                self.current_selected_indices
            )
            episode_stats.append(stats)
            
            # Update current state for NEXT episode/step
            self.current_selected_vertiports = new_selected_vertiports
            self.current_selected_indices = new_selected_indices
            self.current_metrics = simulator_metrics  # ← FIX: Store metrics!
            
            print(f"  Design step {design_step}: "
                  f"Reward={reward:.3f}, "
                  f"Changed={action_changed.sum().item()}/{len(action_changed)} regions")
        
        # Return average stats
        avg_stats = {
            'reward': np.mean(episode_rewards),
            'loss': np.mean([s['loss'] for s in episode_stats]),
            'value_estimate': np.mean([s['value_estimate'] for s in episode_stats]),
            'advantage': np.mean([s['advantage'] for s in episode_stats]),
        }
        
        return episode_rewards, avg_stats
    
    def _collect_simulator_metrics(self, simulator:MapEnv):
        """Collect relevant metrics from simulator"""

        metrics = {
            'vertiport_metrics': {},
            'flow_metrics': {},
        }
        
        # Example metrics - customize based on your simulator
        # This is a placeholder - implement based on your MapEnv
        try:
            metrics['total_throughput'] = getattr(simulator, 'total_throughput', 0.0)
            metrics['avg_wait_time'] = getattr(simulator, 'avg_wait_time', 0.0)
            metrics['avg_delay'] = getattr(simulator, 'avg_delay', 0.0)
        except:
            pass
        
        return metrics


# Main execution
if __name__ == "__main__":
    test_mode = True
    num_regions = 4


    TEST_MODE = True
    
    if TEST_MODE:
        vertiport_tag_list = []
    else:
        vertiport_tag_list = [('building', 'commercial')]
    
    # Initialize simulator
    uam_simulator = MapEnv(
        number_of_uav=0,
        num_ORCA_uav=0,
        number_of_vertiport=100,  # Total candidate vertiports
        location_name='Austin, Texas, USA',
        airspace_tag_list=[],
        vertiport_tag_list=vertiport_tag_list,
        max_episode_steps=100, #TODO: max_step_per_episode
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
    
    #! this attribute is NOT initiated - ERROR attr not defined - FIX:  !!
    uam_simulator.set_airspace_vp_design()
    if test_mode:
        #make vertiports
        #from list of location, or from centeroid
        print('Vertiport Design problem in test mode')
        uam_simulator.airspace.make_regions_dict_vp_des_test_mode()

        #TODO: print the vertiports(with their location) along with their region number 
        #TODO: I want to know what are all the vertiports their regions and each vertiports location 
    # OR use region_tags
    else:
        uam_simulator.airspace.make_regions_dict_vp_des('commercial', num_regions=num_regions)

    # Initialize graph builder
    graph_builder = VertiportGraphBuilder(
        airspace=uam_simulator.airspace,
        node_feature_dim=20,
        edge_feature_dim=12,
        connectivity_type='inter_intra'  # or 'full'
    )
    num_regions = len(uam_simulator.airspace.regions_dict)
    
    # Initialize model
    rl_model = StationSelectionGNNRL(
        node_features=20,
        edge_features=12,
        hidden_dim=128,
        num_regions=num_regions,
        num_gnn_layers=3
    )
    
    # Initialize trainer
    trainer = A2CTrainer(
        model=rl_model,
        graph_builder=graph_builder,
        lr=3e-4,
        gamma=0.99,
        value_coef=0.5,
        entropy_coef=0.01
    )
    
    # Training loop
    num_episodes = 10
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode} ===")
        
        rewards, stats = trainer.train_episode(
            simulator=uam_simulator,
            simulator_steps=3,
            num_design_steps=30  # Number of design iterations per episode
        )
        
        print(f"Episode {episode} Summary:")
        print(f"  Avg Reward: {stats['reward']:.3f}")
        print(f"  Value Estimate: {stats['value_estimate']:.3f}")
        print(f"  Advantage: {stats['advantage']:.3f}")
        print(f"  Loss: {stats['loss']:.3f}")
    
    print(f'TRAINING COMPLETE: vertiport list: {uam_simulator.airspace.vertiport_list}')
    time.sleep(5)
    
    print("\n=== Training Complete ===")
    
    # Inference
    rl_model.eval()
    
    # Build graph with CURRENT state (selection + metrics from training)
    x, edge_index, edge_attr = graph_builder.build_graph(
        selected_vertiports=trainer.current_selected_vertiports,
        metrics=trainer.current_metrics  # Use metrics from training!
    )
    
    selected_indices, value = rl_model.select_vertiports(
        x, edge_index, edge_attr,
        graph_builder.region_mask,
        current_selection=trainer.current_selected_indices,
        deterministic=True
    )
    
    selected_vertiports = graph_builder.indices_to_vertiports(selected_indices)
    
    print(f"\nFinal Selected Vertiports:")
    for i, vp in enumerate(selected_vertiports):
        print(f"  Region {i}: {vp} at ({vp.location.x:.2f}, {vp.location.y:.2f})")
    print(f"Expected Value: {value:.3f}")
    