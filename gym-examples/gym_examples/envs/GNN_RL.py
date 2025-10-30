from stable_baselines3 import PPO
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

class StationGNN(nn.Module):
    """
    Graph Neural Network for station embedding
    Processes the dynamic graph structure where selected stations
    are inter-region connected, non-selected are intra-region connected
    """
    def __init__(self, node_features, edge_features, hidden_dim=128, num_layers=3):
        super(StationGNN, self).__init__()
        
        # Project node and edge features to hidden dimension
        self.node_proj = nn.Linear(node_features, hidden_dim)
        self.edge_proj = nn.Linear(edge_features, hidden_dim)
        
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
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            batch: Batch assignment for nodes (if using batched graphs)
        
        Returns:
            node_embeddings: [num_nodes, hidden_dim]
        """
        # Project inputs
        h = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        # Apply GNN layers with residual connections
        for conv, norm in zip(self.conv_layers, self.layer_norms):
            h_new = conv(h, edge_index, edge_attr)
            h_new = F.relu(h_new)
            h = norm(h + h_new)  # Residual connection
            
        return h #node embedding 
    
class PolicyNetwork(nn.Module):
    """
    Actor: Selects one station per region
    Outputs selection probabilities for each station
    """
    def __init__(self, hidden_dim=128, num_regions=None):
        super(PolicyNetwork, self).__init__()
        
        self.num_regions = num_regions
        
        # MLP for processing node embeddings
        self.policy_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # last layer should have dim = num_node in region - this last layer is the logits which assign probability to each vertiport in that region
                                      # the last layer needs to be flexible - each region has different number of vertiports
                                      # there HAS to be another additional element in the last layer -
                                      #         NO ACTION - meaning, no action,
                                      #         this will not change/update the selected vertiports,
                                      #         since this is a combinatorial problem solved using RL 
        )
        
    def forward(self, node_embeddings, region_mask):
        """
        Args:
            node_embeddings: [num_nodes, hidden_dim]
            region_mask: [num_regions, num_nodes] - indicates which stations belong to which region
        
        Returns:
            action_probs: List of [num_stations_in_region] for each region
            log_probs: Log probabilities of selected actions
            entropy: Entropy of the distribution (for exploration)
        """
        # Compute logits for each station
        logits = self.policy_mlp(node_embeddings).squeeze(-1)  # [num_nodes]
        
        # For each region, apply softmax over stations in that region
        action_probs_list = []
        log_probs_list = []
        entropy_list = []
        selected_stations = []
        
        for region_idx in range(region_mask.shape[0]):
            # Get mask for current region
            mask = region_mask[region_idx]  # [num_nodes]
            region_logits = logits[mask.bool()]  # [num_stations_in_region]
            
            # Softmax over stations in this region
            region_probs = F.softmax(region_logits, dim=0)
            
            # Sample action (station selection)
            dist = torch.distributions.Categorical(region_probs)
            action = dist.sample()
            
            action_probs_list.append(region_probs)
            log_probs_list.append(dist.log_prob(action))
            entropy_list.append(dist.entropy())
            
            # Get global station index
            region_station_indices = torch.where(mask)[0]
            selected_stations.append(region_station_indices[action])
        
        # Stack results
        log_probs = torch.stack(log_probs_list).sum()  # Total log prob
        entropy = torch.stack(entropy_list).mean()  # Average entropy
        selected_stations = torch.stack(selected_stations)  # [num_regions]
        
        return selected_stations, log_probs, entropy


class ValueNetwork(nn.Module):
    """
    Critic: Estimates expected cumulative reward from current state
    
    What it indicates:
    - V(state) = Expected total reward from current station configuration
    - Helps assess: "Is this selection putting us in a good position?"
    
    Why we need it:
    1. Advantage estimation: A(s,a) = R - V(s) reduces variance
    2. Credit assignment: Provides feedback before final reward
    3. Baseline: Stabilizes policy gradient updates
    """
    def __init__(self, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        
        # Global state representation
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Single value estimate
        )
        
    def forward(self, node_embeddings, selected_stations):
        """
        Args:
            node_embeddings: [num_nodes, hidden_dim]
            selected_stations: [num_regions] - indices of selected stations
        
        Returns:
            value: Scalar value estimate for this configuration
        """
        # Aggregate embeddings of selected stations
        selected_embeddings = node_embeddings[selected_stations]  # [num_regions, hidden_dim]
        
        # Global state representation (mean pooling)
        global_state = selected_embeddings.mean(dim=0)  # [hidden_dim]
        
        # Estimate value
        value = self.value_mlp(global_state)
        
        return value.squeeze()


class StationSelectionGNNRL(nn.Module):
    """
    Complete GNN-RL model for station selection
    """
    def __init__(self, node_features, edge_features, hidden_dim=128, 
                 num_regions=None, num_gnn_layers=3):
        super(StationSelectionGNNRL, self).__init__()
        
        self.gnn = StationGNN(node_features, edge_features, hidden_dim, num_gnn_layers)
        self.policy = PolicyNetwork(hidden_dim, num_regions)
        self.value = ValueNetwork(hidden_dim)
        
    def forward(self, x, edge_index, edge_attr, region_mask, batch=None):
        """
        Complete forward pass
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge features
            region_mask: Region assignment for stations
            batch: Batch indices (optional)
        
        Returns:
            selected_stations: [num_regions] - selected station indices
            log_probs: Log probability of the selection
            value: Value estimate for this state
            entropy: Entropy for exploration bonus
        """
        # GNN embedding
        node_embeddings = self.gnn(x, edge_index, edge_attr, batch)
        
        # Policy: Select stations
        selected_stations, log_probs, entropy = self.policy(node_embeddings, region_mask)
        
        # Value: Estimate state value
        value = self.value(node_embeddings, selected_stations)
        
        return selected_stations, log_probs, value, entropy
    
    def get_value(self, x, edge_index, edge_attr, region_mask, selected_stations, batch=None):
        """
        Get value estimate for a given configuration (used during training)
        """
        node_embeddings = self.gnn(x, edge_index, edge_attr, batch)
        value = self.value(node_embeddings, selected_stations)
        return value