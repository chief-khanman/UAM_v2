"""
GNN_RL.py - Graph Neural Network with Reinforcement Learning for Vertiport Selection

This module implements a GNN-RL architecture for combinatorial optimization,
specifically for selecting optimal vertiports from candidate lists in each region.

Key Components:
- StationGNN: Graph Neural Network for embedding vertiports
- PolicyNetwork: Actor that selects vertiports (with NO ACTION option)
- ValueNetwork: Critic that estimates configuration quality
- StationSelectionGNNRL: Combined model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import GATConv


class StationGNN(nn.Module):
    """
    Graph Neural Network for station/vertiport embedding.
    
    Processes the dynamic graph structure where:
    - Selected stations are inter-region connected
    - Non-selected stations are intra-region connected
    
    Uses Graph Attention Networks (GAT) for attention-based message passing.
    """
    
    def __init__(self, node_features, edge_features, hidden_dim=128, num_layers=3):
        """
        Args:
            node_features (int): Number of input node features
            edge_features (int): Number of input edge features
            hidden_dim (int): Hidden dimension for embeddings
            num_layers (int): Number of GNN layers
        """
        super(StationGNN, self).__init__()
        
        #! option 1 
        #  concatenate the selected_node bool vector to the node_features before node_proj
        # Project node and edge features to hidden dimension
        self.node_proj = nn.Linear(node_features, hidden_dim)
        self.edge_proj = nn.Linear(edge_features, hidden_dim)
        
        #! option 2
        #  concatenate the selected_node bool vector to the node_proj 


        # GNN layers (using GAT for attention-based aggregation)
        self.conv_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, edge_dim=hidden_dim, heads=4, concat=False)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
    def forward(self, x, edge_index, edge_attr, batch=None):
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
        # Project inputs to hidden dimension
        h = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        # Apply GNN layers with residual connections
        for conv, norm in zip(self.conv_layers, self.layer_norms):
            h_new = conv(h, edge_index, edge_attr)
            h_new = F.relu(h_new)
            h = norm(h + h_new)  # Residual connection
            
        return h


class PolicyNetwork(nn.Module):
    """
    Actor: Selects one station per region OR chooses NO ACTION.
    
    For combinatorial optimization:
    - Each region can: select a new station OR keep current selection (NO ACTION)
    - This allows the model to make partial updates
    - NO ACTION is implemented as a learnable embedding
    """
    
    def __init__(self, hidden_dim=128, num_regions=None,  shared_no_action = True):
        """
        Args:
            hidden_dim (int): Dimension of node embeddings
            num_regions (int): Number of regions
            shared_no_action (bool): Boolean toggle for shared no_action probability 
        """
        super(PolicyNetwork, self).__init__()
        
        self.shared_no_action = shared_no_action
        self.no_action_region_mask = torch.zeros(num_regions, hidden_dim)
        
        self.num_regions = num_regions
        
        # MLP for processing node embeddings to get selection scores
        self.policy_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Score for each station
        )
        
        # NO ACTION learnable embedding
        # This allows the model to learn when NOT to change selection
        if self.shared_no_action:
            self.no_action_embedding = nn.Parameter(torch.randn(hidden_dim))
        else:
            self.no_action_embedding = nn.Parameter(torch.randn(num_regions, hidden_dim))
        
    def forward(self, node_embeddings, region_mask, current_selection=None, training=True):
        """
        Forward pass to select stations.
        
        Args:
            node_embeddings (torch.Tensor): Node embeddings [num_nodes, hidden_dim]
            region_mask (torch.Tensor): Region assignments [num_regions, num_nodes]
                Binary mask indicating which stations belong to which region
            current_selection (torch.Tensor, optional): Currently selected stations [num_regions]
            training (bool): If True, sample actions; if False, take argmax
        
        Returns:
            selected_stations (torch.Tensor): Selected station indices [num_regions]
            log_probs (torch.Tensor): Log probabilities of selected actions
            entropy (torch.Tensor): Entropy of the distribution (for exploration)
            action_changed (torch.Tensor): Boolean mask of which regions changed [num_regions]
        """
        # Compute logits for each station
        #! option 3 
        #  use a skip connection and concatenate the selected_nodes 
        #TODO: concatenate the selected node boolean vector(skip connection) with node_embedding 
        station_logits = self.policy_mlp(node_embeddings).squeeze(-1)  # [num_nodes]
        if self.shared_no_action: 
            # NO ACTION logit (computed from no_action_embedding)
            no_action_logit = self.policy_mlp(self.no_action_embedding.unsqueeze(0)).squeeze() #scalar number
        else: 
            no_action_logit = self.policy_mlp(self.no_action_embedding).squeeze(-1)  # [num_regions]


        
        action_probs_list = []
        log_probs_list = []
        entropy_list = []
        selected_stations = []
        action_changed = []
        
        for region_idx in range(region_mask.shape[0]):
            # Get mask for current region
            mask = region_mask[region_idx]  # [num_nodes(aka total num vertiports)]
            region_station_logits = station_logits[mask.bool()]  # [num_stations_in_region]
            
            # Add NO ACTION as an option
            # Combined logits: [num_stations_in_region + 1]
            if self.shared_no_action:
                combined_logits = torch.cat([region_station_logits, no_action_logit.unsqueeze(0)])
            else:
                combined_logits = torch.cat([region_station_logits, no_action_logit[region_idx].unsqueeze(0)]) #! should I select the no_action_logit by indexing into it OR should I use the station_mask to select no_action
            
            # Softmax over all options (including NO ACTION)
            region_probs = F.softmax(combined_logits, dim=0)
            
            # Sample or select action
            dist = torch.distributions.Categorical(region_probs)
            if training:
                action = dist.sample() #action is index associated with categorical distribution
            else:
                action = torch.argmax(region_probs)
            
            action_probs_list.append(region_probs)
            log_probs_list.append(dist.log_prob(action))
            entropy_list.append(dist.entropy())
            
            # Determine selected station
            region_station_indices = torch.where(mask)[0] # given [0,0,1,1] returns [2,3] location of 1s in the mask
            num_stations_in_region = len(region_station_indices)
            
            # for a given region lets say there are 4 stations
            # action is index that is  action is one of the indeces 0,1,2,3,4 -> index 4 is no_action/no change in station 
            if action < num_stations_in_region:
                # Selected a station
                selected_station = region_station_indices[action]
                action_changed.append(True)
            else:
                # NO ACTION - keep current selection
                if current_selection is not None:
                    # current_selection -> current_indices 
                    selected_station = current_selection[region_idx] #! example: [vp_id_3(region_0), vp_id_5(region_1), vp_id_10(region_2), vp_id_12(region_3)]
                # else:
                #     # First selection - default to first station in region
                #     #TODO: maybe we can choose a random station to start with 
                #     selected_station = region_station_indices[0]
                action_changed.append(False)
            
            selected_stations.append(selected_station)
        
        # Stack results
        log_probs = torch.stack(log_probs_list).sum()  # Total log prob
        entropy = torch.stack(entropy_list).mean()  # Average entropy
        selected_stations = torch.stack(selected_stations)  # [num_regions]
        action_changed = torch.tensor(action_changed, dtype=torch.bool)  # [num_regions]
        
        return selected_stations, log_probs, entropy, action_changed


class ValueNetwork(nn.Module):
    """
    Critic: Estimates expected reward from current vertiport configuration.
    
    For combinatorial optimization:
    - V(configuration) = Expected quality/reward of this vertiport selection
    - Helps identify which configurations are promising
    - Trained to predict actual rewards from simulator
    
    The value function indicates:
    - How good the current selection is
    - Expected performance before running simulator
    - Used for advantage calculation: A(s,a) = R - V(s)
    """
    
    def __init__(self, hidden_dim=128):
        """
        Args:
            hidden_dim (int): Dimension of node embeddings
        """
        super(ValueNetwork, self).__init__()
        
        # Value estimation MLP
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Single scalar value
        )
    #                 state,           action   
    def forward(self, node_embeddings, selected_stations):
        """
        Forward pass to estimate value.
        
        Args:
            node_embeddings (torch.Tensor): Node embeddings [num_nodes, hidden_dim]
            selected_stations (torch.Tensor): Indices of selected stations [num_regions]
        
        Returns:
            torch.Tensor: Scalar value estimate for this configuration
        """
        # Aggregate embeddings of selected stations
        selected_embeddings = node_embeddings[selected_stations]  # [num_regions, hidden_dim]
        
        # Global state representation (mean pooling)
        # This creates a permutation-invariant representation
        #! is mean a good/correct way to combine all the nodes' information of selected vertiports
        global_state = selected_embeddings.mean(dim=0)  # [hidden_dim]
        
        # Estimate value
        value = self.value_mlp(global_state)
        
        return value.squeeze()


class StationSelectionGNNRL(nn.Module):
    """
    Complete GNN-RL model for vertiport selection (combinatorial optimization).
    
    Combines:
    - StationGNN: Embeds vertiports in graph context
    - PolicyNetwork: Selects vertiports (or NO ACTION)
    - ValueNetwork: Estimates configuration quality
    
    Usage:
        model = StationSelectionGNNRL(
            node_features=20,
            edge_features=12,
            hidden_dim=128,
            num_regions=5,
            num_gnn_layers=3
        )
        
        # Training mode (samples actions)
        selected, log_probs, value, entropy, changed = model(
            x, edge_index, edge_attr, region_mask, 
            current_selection=current, training=True
        )
        
        # Inference mode (deterministic)
        selected, value = model.select_vertiports(
            x, edge_index, edge_attr, region_mask,
            current_selection=current, deterministic=True
        )
    """
    
    def __init__(self, node_features, edge_features, hidden_dim=128, 
                 num_regions=None, num_gnn_layers=3, shared_no_action=True):
        """
        Args:
            node_features (int): Number of node features
            edge_features (int): Number of edge features
            hidden_dim (int): Hidden dimension for embeddings
            num_regions (int): Number of regions
            num_gnn_layers (int): Number of GNN layers
        """
        super(StationSelectionGNNRL, self).__init__()
        
        self.gnn = StationGNN(node_features, edge_features, hidden_dim, num_gnn_layers)
        self.policy = PolicyNetwork(hidden_dim, num_regions,shared_no_action=shared_no_action)
        self.value = ValueNetwork(hidden_dim)
        
    def forward(self, x, edge_index, edge_attr, region_mask, current_selection=None, 
                training=True, batch=None):
        """
        Complete forward pass through the model.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, node_features]
            edge_index (torch.Tensor): Edge connectivity [2, num_edges]
            edge_attr (torch.Tensor): Edge features [num_edges, edge_features]
            region_mask (torch.Tensor): Region assignments [num_regions, num_nodes]
            current_selection (torch.Tensor, optional): Currently selected stations [num_regions]
            training (bool): Whether in training mode (affects sampling vs argmax)
            batch (torch.Tensor, optional): Batch indices for nodes
        
        Returns:
            selected_stations (torch.Tensor): Selected station indices [num_regions]
            log_probs (torch.Tensor): Log probability of the selection
            value (torch.Tensor): Value estimate for this state
            entropy (torch.Tensor): Entropy for exploration bonus
            action_changed (torch.Tensor): Which regions changed selection [num_regions]
        """
        # GNN embedding - only NODE and EDGE features - STATE/OBS 
        node_embeddings = self.gnn(x, edge_index, edge_attr, batch)
        
        # Policy: Select stations (or NO ACTION)
        selected_stations, log_probs, entropy, action_changed = self.policy(
            node_embeddings, region_mask, current_selection, training #! does policy take into account current_selection ??
        )
        
        # Value: Estimate configuration quality
        value = self.value(node_embeddings, selected_stations)
        
        return selected_stations, log_probs, value, entropy, action_changed
    
    def get_value(self, x, edge_index, edge_attr, selected_stations, batch=None):
        """
        Get value estimate for a given configuration (used during training).
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Edge connectivity
            edge_attr (torch.Tensor): Edge features
            selected_stations (torch.Tensor): Indices of selected stations
            batch (torch.Tensor, optional): Batch indices
        
        Returns:
            torch.Tensor: Value estimate
        """
        node_embeddings = self.gnn(x, edge_index, edge_attr, batch)
        value = self.value(node_embeddings, selected_stations)
        return value
    
    def select_vertiports(self, x, edge_index, edge_attr, region_mask, 
                         current_selection=None, deterministic=False):
        """
        Select vertiports for inference (cleaner interface).
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Edge connectivity
            edge_attr (torch.Tensor): Edge features
            region_mask (torch.Tensor): Region assignments
            current_selection (torch.Tensor, optional): Current selection
            deterministic (bool): If True, use argmax instead of sampling
        
        Returns:
            selected_stations (torch.Tensor): Selected station indices
            value (torch.Tensor): Value estimate
        """
        with torch.no_grad():
            selected_stations, _, value, _, _ = self.forward(
                x, edge_index, edge_attr, region_mask, 
                current_selection=current_selection,
                training=not deterministic
            )
        return selected_stations, value


# Example usage and testing
if __name__ == "__main__":
    print("Testing StationSelectionGNNRL...")
    
    # Hyperparameters
    num_nodes = 10
    num_regions = 3
    node_features = 20
    edge_features = 12
    hidden_dim = 128
    
    # Create model
    model = StationSelectionGNNRL(
        node_features=node_features,
        edge_features=edge_features,
        hidden_dim=hidden_dim,
        num_regions=num_regions,
        num_gnn_layers=3
    )
    
    # Create dummy data
    x = torch.randn(num_nodes, node_features)
    edge_index = torch.randint(0, num_nodes, (2, 20))
    edge_attr = torch.randn(20, edge_features)
    
    # Region mask: [num_regions, num_nodes]
    region_mask = torch.zeros(num_regions, num_nodes)
    region_mask[0, :4] = 1  # Region 0: nodes 0-3
    region_mask[1, 4:7] = 1  # Region 1: nodes 4-6
    region_mask[2, 7:] = 1   # Region 2: nodes 7-9
    
    # Forward pass
    selected_stations, log_probs, value, entropy, action_changed = model(
        x, edge_index, edge_attr, region_mask, training=True
    )
    
    print(f"Selected stations: {selected_stations}")
    print(f"Log probabilities: {log_probs.item():.4f}")
    print(f"Value estimate: {value.item():.4f}")
    print(f"Entropy: {entropy.item():.4f}")
    print(f"Actions changed: {action_changed}")
    
    # Test with current selection
    selected_stations_2, log_probs_2, value_2, entropy_2, action_changed_2 = model(
        x, edge_index, edge_attr, region_mask, 
        current_selection=selected_stations,
        training=True
    )
    
    print(f"\nSecond selection: {selected_stations_2}")
    print(f"Actions changed: {action_changed_2}")
    
    # Test inference mode
    selected_stations_inf, value_inf = model.select_vertiports(
        x, edge_index, edge_attr, region_mask,
        current_selection=selected_stations,
        deterministic=True
    )
    
    print(f"\nInference (deterministic): {selected_stations_inf}")
    print(f"Value: {value_inf.item():.4f}")
    
    print("\nModel test complete!")