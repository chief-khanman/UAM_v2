import torch
import torch.nn.functional as F
from GNN_RL import StationSelectionGNNRL
from map_env_revised import MapEnv
class A2CTrainer:
    """
    Advantage Actor-Critic (A2C) trainer
    
    Value Network Role in Training:
    1. Computes advantage: advantage = reward - value_estimate
    2. Advantage tells us if the action was better/worse than expected
    3. This reduces variance and speeds up learning
    """
    def __init__(self, model, lr=3e-4, gamma=0.99, value_coef=0.5, entropy_coef=0.01):
        self.model = model # StationSelectionGNNRL instance
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
    def train_step(self, 
                   graph_data, 
                   region_mask, 
                   reward):
        """
        Single training step
        
        Args:
            graph_data: (x, edge_index, edge_attr) for the current graph used by simulator
            region_mask: Region assignments
            reward: Reward from simulator after station selection
        """
        x, edge_index, edge_attr = graph_data
        
        # Forward pass
        selected_stations, log_probs, value, entropy = self.model(
            x, edge_index, edge_attr, region_mask
        )
        
        # Compute advantage
        # This is where the value network is crucial!
        
        # reward comes from simulator - GNN_RL returns reward
        advantage = reward - value.detach()  # How much better than expected?
        
        # Policy loss (Actor)
        # We want to increase probability of actions with positive advantage
        policy_loss = -(log_probs * advantage)
        
        # Value loss (Critic)
        # Train value network to predict actual rewards
        value_loss = F.mse_loss(value, torch.tensor(reward))
        
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
            'advantage': advantage.item()
        }
    
    def train_episode(self, 
                      simulator:MapEnv, 
                      simulator_steps , 
                      initial_graph, 
                      region_mask, 
                      num_steps=1): #this step is for running GNN_RL vertiport design problem only once per episode - we can change this to more steps if needed
        """
        Train for one episode
        
        In your case with single-step episodes:
        1. Model selects stations
        2. Simulator runs and returns reward
        3. Update model using reward and value estimate
        """
        episode_rewards = []
        current_graph = initial_graph # this will have the graph and structure - the node and edge information will need to be updated
        
        for step in range(num_steps): # number of times we will run GNN_RL vertiport design problem in one episode

            #stages of the episode
            # 1. reset GNN_RL - this will provide with selected vertiports 
            # 2. create a graph with the selected vertiports based on graph connection type option a)full connected graph b) inter-intra connection
            # 3. run simulator to collect simulator_initial_metrics, and simulator_end_metrics
            # 4. use these metrics to build and complete the node and edge features
            # 5. reward from the simulator will be transformed to assign as reward for the vertiport design problem
            #    

            # Run simulator with selected configuration
            vertiport_selected_by_gnnrl = None #! convert selected_stations to vertiport list
            vertiport_for_mapped_env = vertiport_selected_by_gnnrl
            simulator.airspace.set_vertiport_list_vp_design(vertiport_for_mapped_env)
            map_env_obs, map_env_info = simulator.reset(seed=None) #! need to provide a seed
            
            initial_metrics = simulator._collect_initial_metrics()
            for sim_step in range(simulator_steps):
                map_env_obs, map_env_reward, map_env_terminated, map_env_truncated, map_env_info = simulator.step(simulator.action_space.sample()) #! this is the map_env - need to apply another for loop around this to run uam_simulator.step()
            
            end_metrics = simulator._collect_episode_end_metrics()
            reward = self._get_reward(initial_metrics, end_metrics)
            episode_rewards.append(reward)


            #! need to define a set_graph_state() - which will use initial_metrics and end_metrics 
            #! BOTH set and get method needs to be defined 
            # set_graph_state(), will use initial_metrics, and end_metrics use information from those to update/add node/edge features
            self.set_graph_state(initial_metrics, end_metrics)

            
            # Get current graph state
            x, edge_index, edge_attr = self.get_graph_state(current_graph) # this method will extract node features, edge index, edge attributes from the graph
            
            # Model selects stations
            with torch.no_grad():
                new_selected_stations, _, _, _ = self.model(
                    x, edge_index, edge_attr, region_mask
                )
            
            # Update graph: set selected stations and update edges
            current_graph = self.update_graph_structure(
                initial_graph, new_selected_stations, region_mask
            )
            
            
            # Training step
            stats = self.train_step(
                (x, edge_index, edge_attr), 
                region_mask, 
                reward
            )
            
        return episode_rewards, stats
    
    
    def get_graph_state(self,):
        pass


    def set_graph_state(self,):
        pass



    #! this needs to be fixed - for the first trial run reward will only be sum of distance between vertiports
    def _get_reward(self, initial_metrics, end_metrics):
        """
        Compute reward based on initial and end metrics from simulator
        Customize this based on your reward structure
        """
        # Example: reward = improvement in some metric
        initial_value = None
        end_value = None
        reward = end_value + initial_value
        return reward
    




    # this method will need some update - there can be two ways a graph can be made 1) fully connected graph, 2) inter-intra region graph
    #define a new methods that will build the graph using regions, and selected vertiports
    def update_graph_structure(self, graph, selected_stations, region_mask):
        """
        Update edge structure based on selection:
        - Selected stations: connected to all other selected stations
        - Non-selected: connected only within region
        """
        num_nodes = graph['x'].shape[0]
        num_regions = region_mask.shape[0]
        
        new_edges = []
        new_edge_attrs = []
        
        # Inter-region edges (between selected stations)
        for i in range(num_regions):
            for j in range(i + 1, num_regions):
                station_i = selected_stations[i]
                station_j = selected_stations[j]
                
                # Add bidirectional edge
                new_edges.append([station_i, station_j])
                new_edges.append([station_j, station_i])
                
                # Inter-region edge features
                edge_feat = self.compute_edge_features(station_i, station_j, 'inter')
                new_edge_attrs.extend([edge_feat, edge_feat])
        
        # Intra-region edges (within each region, all stations)
        for region_idx in range(num_regions):
            region_stations = torch.where(region_mask[region_idx])[0]
            
            for i in range(len(region_stations)):
                for j in range(i + 1, len(region_stations)):
                    station_i = region_stations[i]
                    station_j = region_stations[j]
                    
                    new_edges.append([station_i, station_j])
                    new_edges.append([station_j, station_i])
                    
                    edge_feat = self.compute_edge_features(station_i, station_j, 'intra')
                    new_edge_attrs.extend([edge_feat, edge_feat])
        
        edge_index = torch.tensor(new_edges).t()
        edge_attr = torch.stack(new_edge_attrs)
        
        return {
            'x': graph['x'],
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }
    
    def compute_edge_features(self, station_i, station_j, edge_type):
        """
        Compute edge features between two stations
        Can include: distance, capacity, historical flow, etc.
        """
        # Placeholder - implement based on your problem
        base_features = torch.randn(10)  # Example edge features
        
        if edge_type == 'inter':
            type_encoding = torch.tensor([1.0, 0.0])
        else:
            type_encoding = torch.tensor([0.0, 1.0])
        
        return torch.cat([base_features, type_encoding])
    
TEST_MODE = True

if TEST_MODE:
    vertiport_tag_list = []
else:
    vertiport_tag_list=[('building', 'commercial')]

uam_simulator = MapEnv(number_of_uav=0,
                   num_ORCA_uav=0,
                   number_of_vertiport=2, #! this argument is ONLY used for creating random vertiports 
                   location_name='Austin, Texas, USA',
                   airspace_tag_list=[],
                   vertiport_tag_list=vertiport_tag_list, #! use a tag-tag_str that has few vertiports  
                   max_episode_steps=100,
                   number_of_other_agents_for_model=7,#what is this???
                   sleep_time=0,
                   seed=70,
                   obs_space_str='UAM_UAV',
                   sorting_criteria='closest_first',
                   render_mode=None,
                   max_uavs=100,
                   max_vertiports=150,
                   make_uav_at_timestep=300,
                   vp_design_problem=True)



# Initialize model
rl_algo_model = StationSelectionGNNRL(
    node_features=20,      # e.g., demand, capacity, location, etc.
    edge_features=12,      # distance, flow capacity, etc.
    hidden_dim=128,
    num_regions=5,
    num_gnn_layers=3
)

# Initialize trainer
trainer = A2CTrainer(
    model=rl_algo_model,
    lr=3e-4,
    gamma=0.99,          # Discount factor
    value_coef=0.5,      # Weight for value loss
    entropy_coef=0.01    # Weight for exploration
)

##### ------ Define these functions ----- #####

#TODO: create_initial_graph and create_region_mask functions
def create_initial_graph():
    # HOW to make initial graph -
    # create graph where nodes are all vertiports in the airspace
    # connect nodes based on full graph connectivity OR inter-intra connectivity(this is an argument)  
    pass

def create_region_mask():
    # HOW to create region mask - 
    # airspace attribute - airspace.region_dict is keyed sequentially with regions starting at 0
    # for each region in region_dict returns vertiport list of that region
    # place all vertiports in a temporary variable
    # keep track of vertiports in each regions and create corresponding mask    
    pass

##### ------ #####


#### MAIN TRAINING LOOP ####

num_episodes = 100
# Training loop
for episode in range(num_episodes):
    # Initialize graph with all stations
    initial_graph = create_initial_graph()
    region_mask = create_region_mask()  # [num_regions, num_stations]
    
    # Train episode
    rewards, stats = trainer.train_episode(
        simulator=uam_simulator,
        simulator_steps=100,  # Steps to run simulator after station selection
        initial_graph=initial_graph, #graph made from current selected vertiports
        region_mask=region_mask,
        num_steps=1          # Single-step episode
    )
    
    # Logging
    print(f"Episode {episode}")
    print(f"  Reward: {stats['reward']:.3f}")
    print(f"  Value Estimate: {stats['value_estimate']:.3f}")
    print(f"  Advantage: {stats['advantage']:.3f}")
    print(f"  Value Loss: {stats['value_loss']:.3f}")


#### TRAINING COMPLETE - INFERENCE ####

# Inference (after training)
rl_algo_model.eval()

with torch.no_grad():
    x, edge_index, edge_attr, region_mask = None, None, None, None  # Get from current graph state

    selected_stations, _, value, _ = rl_algo_model(
        x, edge_index, edge_attr, region_mask
    )
    print(f"Selected stations: {selected_stations}")
    print(f"Expected reward: {value:.3f}")