import time
import torch
import torch.nn.functional as F
import numpy as np
from shapely.geometry import Point
from GNN_RL import StationSelectionGNNRL
from map_env_revised import MapEnv
from vertiport import Vertiport

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from collections import defaultdict, deque
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class MetricsTracker:
    """
    Tracks and visualizes all training metrics
    """
    def __init__(self, hidden_dim, shared_no_action, save_dir='./GNN_RL_training_results', use_tensorboard=True):
        """
        Args:
            save_dir: Directory to save plots and logs
            use_tensorboard: Whether to use TensorBoard logging
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        #hidden dim current experiment
        self.hidden_dim = hidden_dim
        #shared_action bool current experiment 
        self.shared_no_action = shared_no_action

        # Episode-level metrics
        self.episode_metrics = defaultdict(list)
        
        # Step-level metrics (within episodes)
        self.step_metrics = defaultdict(list)
        
        # Best performance tracking
        self.best_reward = float('-inf')
        self.best_configuration = None
        
        # Moving averages
        self.reward_window = deque(maxlen=10)
        
        # TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            tb_dir = os.path.join(save_dir, f'tensorboard_{datetime.now().strftime("%Y%m%d_%H%M")}')
            self.writer = SummaryWriter(tb_dir)
            print(f"TensorBoard logging to: {tb_dir}")
        
        # Real-time plotting
        self.fig = None
        self.axes = None
        self.setup_realtime_plot()
    
    def setup_realtime_plot(self):
        """Setup real-time plotting window"""
        plt.ion()  # Interactive mode
        self.fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)
        
        self.axes = {
            'reward': self.fig.add_subplot(gs[0, 0]),
            'loss': self.fig.add_subplot(gs[0, 1]),
            'value': self.fig.add_subplot(gs[0, 2]),
            'advantage': self.fig.add_subplot(gs[1, 0]),
            'policy_loss': self.fig.add_subplot(gs[1, 1]),
            'value_loss': self.fig.add_subplot(gs[1, 2]),
            'distance': self.fig.add_subplot(gs[2, 0]),
            'changes': self.fig.add_subplot(gs[2, 1]),
            'moving_avg': self.fig.add_subplot(gs[2, 2])
        }
        
        plt.show(block=False)
    #! why is episode_rewards unused
    def log_episode(self, episode, episode_rewards, stats, design_step_data):
        """
        Log metrics for a completed episode
        
        Args:
            episode: Episode number
            episode_rewards: List of rewards for each design step
            stats: Aggregated statistics for the episode
            design_step_data: Detailed data for each design step
        """
        # Episode-level metrics
        self.episode_metrics['episode'].append(episode)
        self.episode_metrics['avg_reward'].append(stats['reward'])
        self.episode_metrics['avg_loss'].append(stats['loss'])
        self.episode_metrics['avg_value'].append(stats['value_estimate'])
        self.episode_metrics['avg_advantage'].append(stats['advantage'])
        self.episode_metrics['avg_policy_loss'].append(stats['avg_policy_loss'])
        self.episode_metrics['avg_value_loss'].append(stats['avg_value_loss'])
        
        # Additional episode metrics
        if 'total_distance' in stats:
            self.episode_metrics['total_distance'].append(stats['total_distance'])
        if 'num_changes' in stats:
            self.episode_metrics['num_changes'].append(stats['num_changes'])
        
        # Moving average
        self.reward_window.append(stats['reward'])
        self.episode_metrics['reward_ma'].append(np.mean(self.reward_window))
        
        # Track best performance
        if stats['reward'] > self.best_reward:
            self.best_reward = stats['reward']
            if 'selected_vertiports' in stats:
                self.best_configuration = stats['selected_vertiports']
        
        # Step-level metrics
        for step_idx, step_data in enumerate(design_step_data):
            self.step_metrics['episode'].append(episode)
            self.step_metrics['step'].append(step_idx)
            self.step_metrics['reward'].append(step_data['reward'])
            self.step_metrics['loss'].append(step_data['loss'])
            self.step_metrics['policy_loss'].append(step_data['policy_loss'])
            self.step_metrics['value_loss'].append(step_data['value_loss'])
            self.step_metrics['value_estimate'].append(step_data['value_estimate'])
            self.step_metrics['advantage'].append(step_data['advantage'])
            
            if 'entropy' in step_data:
                self.step_metrics['entropy'].append(step_data['entropy'])
            if 'action_changed_count' in step_data:
                self.step_metrics['action_changed'].append(step_data['action_changed_count'])
        
        # TensorBoard logging
        if self.use_tensorboard:
            self.writer.add_scalar('Episode/Reward', stats['reward'], episode)
            self.writer.add_scalar('Episode/Loss', stats['loss'], episode)
            self.writer.add_scalar('Episode/Value', stats['value_estimate'], episode)
            self.writer.add_scalar('Episode/Advantage', stats['advantage'], episode)
            self.writer.add_scalar('Episode/RewardMovingAvg', np.mean(self.reward_window), episode)
            
            if 'total_distance' in stats:
                self.writer.add_scalar('Episode/TotalDistance', stats['total_distance'], episode)
            if 'num_changes' in stats:
                self.writer.add_scalar('Episode/NumChanges', stats['num_changes'], episode)
        
        # Update real-time plots
        self.update_realtime_plot()
    
    def update_realtime_plot(self):
        """Update real-time plots with latest data"""
        # this conditional is used as a check statement/sentinal - NOW is this sentinal necessary 
        if len(self.episode_metrics['episode']) == 0:
            return
        
        episodes = self.episode_metrics['episode']
        self.fig.suptitle(f'Training- hidden dim:{self.hidden_dim}, shared_no_action:{self.shared_no_action} total_eps: {self.episode_metrics["episode"][-1]} steps_per_ep:{len(self.step_metrics)}')
        # Clear all axes
        for ax in self.axes.values():
            ax.clear()
        
        # 1. Average Reward per Episode
        ax = self.axes['reward']
        ax.plot(episodes, self.episode_metrics['avg_reward'], 'b-', linewidth=2, label='Avg Reward')
        ax.axhline(y=self.best_reward, color='r', linestyle='--', alpha=0.5, label=f'Best: {self.best_reward:.2f}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Average Reward per Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Loss
        ax = self.axes['loss']
        ax.plot(episodes, self.episode_metrics['avg_loss'], 'r-', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title('Average Total Loss')
        ax.grid(True, alpha=0.3)
        
        # 3. Value Estimate
        ax = self.axes['value']
        ax.plot(episodes, self.episode_metrics['avg_value'], 'g-', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.set_title('Average Value Estimate')
        ax.grid(True, alpha=0.3)
        
        # 4. Advantage
        ax = self.axes['advantage']
        ax.plot(episodes, self.episode_metrics['avg_advantage'], 'purple', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Advantage')
        ax.set_title('Average Advantage')
        ax.grid(True, alpha=0.3)
        
        # 5. Policy Loss
        ax = self.axes['policy_loss']
        if 'avg_policy_loss' in self.episode_metrics and len(self.episode_metrics['avg_policy_loss']) > 0:
            ax.plot(episodes, self.episode_metrics['avg_policy_loss'], 'orange', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Average Policy Loss')
        ax.grid(True, alpha=0.3)
        
        # 6. Value Loss
        ax = self.axes['value_loss']
        if 'avg_value_loss' in self.episode_metrics and len(self.episode_metrics['avg_value_loss']) > 0:
            ax.plot(episodes, self.episode_metrics['avg_value_loss'], 'brown', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value Loss')
        ax.set_title('Average Value Loss')
        ax.grid(True, alpha=0.3)
        
        # 7. Total Distance
        ax = self.axes['distance']
        if 'total_distance' in self.episode_metrics and len(self.episode_metrics['total_distance']) > 0:
            ax.plot(episodes, self.episode_metrics['total_distance'], 'cyan', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Distance')
        ax.set_title('Total Pairwise Distance')
        ax.grid(True, alpha=0.3)
        
        # 8. Vertiport Changes
        ax = self.axes['changes']
        if 'num_changes' in self.episode_metrics and len(self.episode_metrics['num_changes']) > 0:
            ax.plot(episodes, self.episode_metrics['num_changes'], 'magenta', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Number of Changes')
        ax.set_title('Vertiport Changes per Episode')
        ax.grid(True, alpha=0.3)
        
        # 9. Moving Average
        ax = self.axes['moving_avg']
        ax.plot(episodes, self.episode_metrics['avg_reward'], 'b-', alpha=0.3, label='Raw Reward')
        ax.plot(episodes, self.episode_metrics['reward_ma'], 'b-', linewidth=2, label='10-Episode MA')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Moving Average')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.pause(0.001)  # Brief pause to update display
    
    def save_all_plots(self):
        """Save all plots to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the main real-time plot
        if self.fig is not None:
            main_plot_path = os.path.join(self.save_dir, f'training_progress_{timestamp}.png')
            self.fig.savefig(main_plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved main plot to: {main_plot_path}")
        
        # Create and save detailed plots
        self._save_detailed_plots(timestamp)
        
        # Create and save summary dashboard
        self._save_summary_dashboard(timestamp)
    
    def _save_detailed_plots(self, timestamp):
        """Save detailed individual plots"""
        # Rewards over steps
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training- hidden dim:{self.hidden_dim}, shared_no_action:{self.shared_no_action} total_eps: {self.episode_metrics["episode"][-1]} steps_per_ep:{len(self.step_metrics)}')
        
        # Step-level rewards
        if len(self.step_metrics['reward']) > 0:
            axes[0, 0].scatter(range(len(self.step_metrics['reward'])), 
                             self.step_metrics['reward'], alpha=0.5, s=10)
            axes[0, 0].set_title('Reward per Design Step (All Episodes)')
            axes[0, 0].set_xlabel('Global Step')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Loss components
        if len(self.step_metrics['policy_loss']) > 0:
            axes[0, 1].plot(self.step_metrics['policy_loss'], label='Policy Loss', alpha=0.7)
            axes[0, 1].plot(self.step_metrics['value_loss'], label='Value Loss', alpha=0.7)
            axes[0, 1].set_title('Loss Components over Steps')
            axes[0, 1].set_xlabel('Global Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Entropy (if available)
        if 'entropy' in self.step_metrics and len(self.step_metrics['entropy']) > 0:
            axes[1, 0].plot(self.step_metrics['entropy'], color='purple', alpha=0.7)
            axes[1, 0].set_title('Entropy over Steps')
            axes[1, 0].set_xlabel('Global Step')
            axes[1, 0].set_ylabel('Entropy')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Action changes (if available)
        if 'action_changed' in self.step_metrics and len(self.step_metrics['action_changed']) > 0:
            axes[1, 1].scatter(range(len(self.step_metrics['action_changed'])), 
                             self.step_metrics['action_changed'], alpha=0.5, s=10, color='green')
            axes[1, 1].set_title('Regions Changed per Step')
            axes[1, 1].set_xlabel('Global Step')
            axes[1, 1].set_ylabel('Number of Regions Changed')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        detailed_path = os.path.join(self.save_dir, f'detailed_metrics_{timestamp}.png')
        fig.savefig(detailed_path, dpi=300, bbox_inches='tight')
        print(f"Saved detailed plots to: {detailed_path}")
        plt.close(fig)
    
    def _save_summary_dashboard(self, timestamp):
        """Create and save comprehensive summary dashboard"""
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'Training- hidden dim:{self.hidden_dim}, shared_no_action:{self.shared_no_action} total_eps: {self.episode_metrics["episode"][-1]} steps_per_ep:{len(self.step_metrics)}')
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        episodes = self.episode_metrics['episode']
        
        # 1. Reward with confidence bands
        ax1 = fig.add_subplot(gs[0, :2])
        if len(episodes) > 0:
            ax1.plot(episodes, self.episode_metrics['avg_reward'], 'b-', linewidth=2, label='Avg Reward')
            ax1.plot(episodes, self.episode_metrics['reward_ma'], 'r-', linewidth=2, label='Moving Avg')
            ax1.fill_between(episodes, 
                           np.array(self.episode_metrics['avg_reward']) - np.std(self.episode_metrics['avg_reward']),
                           np.array(self.episode_metrics['avg_reward']) + np.std(self.episode_metrics['avg_reward']),
                           alpha=0.2)
            ax1.axhline(y=self.best_reward, color='g', linestyle='--', label=f'Best: {self.best_reward:.2f}')
            ax1.set_xlabel('Episode', fontsize=12)
            ax1.set_ylabel('Reward', fontsize=12)
            ax1.set_title('Training Reward Progress', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
        
        # 2. Loss breakdown
        ax2 = fig.add_subplot(gs[0, 2:])
        if len(episodes) > 0:
            ax2.plot(episodes, self.episode_metrics['avg_loss'], label='Total Loss', linewidth=2)
            ax2.set_xlabel('Episode', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.set_title('Training Loss', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        # 3. Value and Advantage
        ax3 = fig.add_subplot(gs[1, 0:2])
        if len(episodes) > 0:
            ax3_twin = ax3.twinx()
            ax3.plot(episodes, self.episode_metrics['avg_value'], 'g-', linewidth=2, label='Value Estimate')
            ax3_twin.plot(episodes, self.episode_metrics['avg_advantage'], 'orange', linewidth=2, label='Advantage')
            ax3.set_xlabel('Episode', fontsize=12)
            ax3.set_ylabel('Value Estimate', fontsize=12, color='g')
            ax3_twin.set_ylabel('Advantage', fontsize=12, color='orange')
            ax3.set_title('Value Estimates and Advantage', fontsize=14, fontweight='bold')
            ax3.tick_params(axis='y', labelcolor='g')
            ax3_twin.tick_params(axis='y', labelcolor='orange')
            ax3.grid(True, alpha=0.3)
        
        # 4. Distribution of rewards
        ax4 = fig.add_subplot(gs[1, 2:])
        if len(self.step_metrics['reward']) > 0:
            ax4.hist(self.step_metrics['reward'], bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax4.axvline(x=np.mean(self.step_metrics['reward']), color='r', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(self.step_metrics["reward"]):.2f}')
            ax4.set_xlabel('Reward', fontsize=12)
            ax4.set_ylabel('Frequency', fontsize=12)
            ax4.set_title('Reward Distribution (All Steps)', fontsize=14, fontweight='bold')
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Learning progress heatmap
        ax5 = fig.add_subplot(gs[2, :2])
        if len(self.step_metrics['episode']) > 0 and len(self.step_metrics['reward']) > 0:
            # Create episode x step reward matrix
            max_episode = max(self.step_metrics['episode'])
            max_step = max(self.step_metrics['step'])
            reward_matrix = np.zeros((max_episode + 1, max_step + 1))
            
            for ep, step, reward in zip(self.step_metrics['episode'], 
                                       self.step_metrics['step'], 
                                       self.step_metrics['reward']):
                reward_matrix[ep, step] = reward
            
            im = ax5.imshow(reward_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest')
            ax5.set_xlabel('Design Step', fontsize=12)
            ax5.set_ylabel('Episode', fontsize=12)
            ax5.set_title('Reward Heatmap (Episode x Step)', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax5, label='Reward')
        
        # 6. Performance metrics summary
        ax6 = fig.add_subplot(gs[2, 2:])
        ax6.axis('off')
        
        summary_text = f"""
        TRAINING SUMMARY
        {'='*50}
        
        Total Episodes: {len(episodes)}
        Total Steps: {len(self.step_metrics['reward'])}
        
        REWARD STATISTICS
        Best Reward: {self.best_reward:.4f}
        Final Reward: {self.episode_metrics['avg_reward'][-1]:.4f}
        Mean Reward: {np.mean(self.episode_metrics['avg_reward']):.4f}
        Std Reward: {np.std(self.episode_metrics['avg_reward']):.4f}
        
        LOSS STATISTICS
        Final Loss: {self.episode_metrics['avg_loss'][-1]:.4f}
        Mean Loss: {np.mean(self.episode_metrics['avg_loss']):.4f}
        
        VALUE ESTIMATES
        Final Value: {self.episode_metrics['avg_value'][-1]:.4f}
        Mean Value: {np.mean(self.episode_metrics['avg_value']):.4f}
        
        ADVANTAGE
        Final Advantage: {self.episode_metrics['avg_advantage'][-1]:.4f}
        Mean Advantage: {np.mean(self.episode_metrics['avg_advantage']):.4f}
        """
        
        ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('COMPREHENSIVE TRAINING DASHBOARD', fontsize=16, fontweight='bold', y=0.995)
        
        dashboard_path = os.path.join(self.save_dir, f'summary_dashboard_{timestamp}.png')
        fig.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary dashboard to: {dashboard_path}")
        plt.close(fig)
    
    def print_summary(self):
        """Print training summary to console"""
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        print(f"Total Episodes: {len(self.episode_metrics['episode'])}")
        print(f"Total Steps: {len(self.step_metrics['reward'])}")
        print(f"\nBest Reward Achieved: {self.best_reward:.4f}")
        print(f"Final Episode Reward: {self.episode_metrics['avg_reward'][-1]:.4f}")
        print(f"Mean Episode Reward: {np.mean(self.episode_metrics['avg_reward']):.4f} ± {np.std(self.episode_metrics['avg_reward']):.4f}")
        print(f"\nFinal Loss: {self.episode_metrics['avg_loss'][-1]:.4f}")
        print(f"Mean Loss: {np.mean(self.episode_metrics['avg_loss']):.4f}")
        print(f"\nFinal Value Estimate: {self.episode_metrics['avg_value'][-1]:.4f}")
        print(f"Mean Value Estimate: {np.mean(self.episode_metrics['avg_value']):.4f}")
        
        if self.best_configuration:
            print(f"\nBest Configuration:")
            for i, vp in enumerate(self.best_configuration):
                print(f"  Region {i}: {vp}")
        
        print("="*70 + "\n")
    
    def close(self):
        """Clean up resources"""
        if self.use_tensorboard:
            self.writer.close()
        
        plt.close('all')


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
    #TODO: Add new feature - binary variable for selected vertiports 
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


class A2CTrainer:
    """
    Advantage Actor-Critic trainer for vertiport selection (combinatorial optimization)
    """
    def __init__(self, model, graph_builder, lr=3e-4, gamma=0.99, 
                 value_coef=0.5, entropy_coef=0.01, metrics_tracker=None):
        self.model = model
        self.graph_builder:VertiportGraphBuilder = graph_builder
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Metrics tracker
        self.metrics_tracker = metrics_tracker
        
        # Track current selection AND metrics (STATE = config + metrics)
        self.current_selected_vertiports = None
        self.current_selected_indices = None
        self.current_metrics = None  # Store metrics as part of state!
        
    #TODO: redefine reward - reward should be the difference between distances of old_vertiports and new_vertiports
    #                  args: new_selected_vertiports, current_selected_vertiports, new_selected_metrics, current_selected_metrics
    #! need to ensure order of subtraction (new - old) OR (old - new), this will have an impact on learning/training 
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
    
    def train_step(self, new_graph_data, current_value, current_log_prob, current_entropy, region_mask, reward, new_selection, new_selected_vertiports,action_changed):
        """
        Single training step
        
        Args:
            new_graph_data: (x, edge_index, edge_attr) for the graph
            current_value: Value of being in current_state(value determined from current_metrics) 
            current_log_prob
            current_entropy
            region_mask: Region assignments
            reward: Reward for current configuration
            new_selection: New selected vertiport indices
            new_selected_vertiports
            action_changed
        """
        x, edge_index, edge_attr = new_graph_data # new_state, s'
        
        # Forward pass
        # selected_station == a''
        #! should the entropy be of the new_state, OR the old_state
        # selected_stations, log_probs, new_value, entropy, action_changed = self.model(
        #     x, edge_index, edge_attr, region_mask, 
        #     current_selection=new_selection,
        #     training=True
        # )

        with torch.no_grad():
            new_value = self.model.get_value(x, edge_index, edge_attr, new_selection)
        
        
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        
        
        td_target = reward_tensor + self.gamma * new_value

        td_target = td_target.detach()
        # Compute advantage
        #advantage = reward_tensor - value.detach() 
        #! why is advantage reward - value, should this be one step TD, reward + gamma*value(s') - value(s)
        advantage =td_target - current_value.detach()
        
        # Policy loss (Actor)
        policy_loss = -(current_log_prob * advantage.detach()) # def: alpha * grad(log(prob_action)) * adv -> should this be the policy loss
        
        # Value loss (Critic)
        #value_loss = F.mse_loss(value, reward_tensor) 
        #! is this correct - should this be one step TD error as well -  value(s), reward + gamma*value(s')
        value_loss = F.mse_loss(current_value, td_target)
        
        # Entropy bonus (for exploration)
        entropy_loss = -current_entropy
        
        # Combined loss
        loss = (policy_loss + #where is the alpha for the policy loss 
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
            'entropy': current_entropy.item(),  # Added entropy to return dict
            'value_estimate': new_value.item(),
            'reward': reward,
            'advantage': advantage.item(),
            'selected_stations': new_selected_vertiports,
            'action_changed_count': action_changed.sum().item()  # Added action change count
        }
    
    def train_episode(self, simulator:MapEnv, simulator_steps, num_design_steps=1, episode_num=0):
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
            episode_num: Current episode number (for logging)
        """
        episode_rewards = []
        episode_stats = []
        episode_distances = []
        episode_changes = []
        
        # Initialize with first vertiport from each region if no current selection
        if self.current_selected_vertiports is None:
            self.current_selected_vertiports = []
            for region_id in sorted(self.graph_builder.airspace.regions_dict.keys()):
                first_vp = self.graph_builder.airspace.regions_dict[region_id][0]
                self.current_selected_vertiports.append(first_vp)
            self.current_selected_indices = self.graph_builder.vertiports_to_indices(
                self.current_selected_vertiports
            )
        
        # step(s) in episode  
        for design_step in range(num_design_steps):
            # Build graph with current selection AND PREVIOUS METRICS
            # State defined using 'current' airspace metrics 
            #! STATE, s 
            x, edge_index, edge_attr = self.graph_builder.build_graph(
                selected_vertiports=self.current_selected_vertiports,
                metrics=self.current_metrics
            )
            
            # Model proposes new selection (or keeps current)
            # Given current state -> returns action (new vertiport selection)
            #! ACTION
            # with torch.no_grad():
            new_selected_indices, current_log_prob, current_value, current_entropy, action_changed = self.model(
                x, edge_index, edge_attr, 
                self.graph_builder.region_mask,
                self.current_selected_indices,
                training=True
            )
            #! ACTION
            # Convert to vertiports
            new_selected_vertiports = self.graph_builder.indices_to_vertiports(
                new_selected_indices
            )
            #! ACTION
            # Set vertiports in simulator
            simulator.airspace.set_vertiport_list_vp_design(new_selected_vertiports)
            #! ENV
            # Run simulator
            obs, info = simulator.reset(seed=None)
            
            for sim_step in range(simulator_steps):
                obs, sim_reward, terminated, truncated, info = simulator.step(
                    simulator.action_space.sample()
                )
                if terminated or truncated:
                    break
            
            # Collect metrics from simulator
            #! NEW_STATE, s'
            simulator_metrics = self._collect_simulator_metrics(simulator)
            
            # Compute reward for this NEW configuration
            #! REWARD 
            reward, total_distance = self.compute_reward(new_selected_vertiports, simulator_metrics) #! should the reward be the difference between previous state total distance and new state total distance 
            
            episode_rewards.append(reward)
            episode_distances.append(total_distance)
            episode_changes.append(action_changed.sum().item())
            
            # Rebuild graph with NEW selection and NEW metrics
            #! NEW STATE, s' -> defined using 'new' airspace metrics
            x_new, edge_index_new, edge_attr_new = self.graph_builder.build_graph(
                selected_vertiports=new_selected_vertiports,
                metrics=simulator_metrics
            )
            

            # Training step: learn from transition
            stats = self.train_step(
                (x_new, edge_index_new, edge_attr_new), #new_state - need for new_value
                current_value, #current_value
                current_log_prob, #current_log_prob
                current_entropy,
                self.graph_builder.region_mask,
                reward,
                new_selected_indices, # new_selected_indices - need for new_value
                new_selected_vertiports,
                action_changed
            )
            episode_stats.append(stats)
            
            # Update current state for NEXT episode/step
            self.current_selected_vertiports = new_selected_vertiports #action
            self.current_selected_indices = new_selected_indices #action
            self.current_metrics = simulator_metrics # new_state, S'
            
            print(f"  Design step {design_step}: "
                  f"Reward={reward:.3f}, "
                  f"Distance={total_distance:.2f}, "
                  f"Changed={action_changed.sum().item()}/{len(action_changed)} regions")
        
        # Compute average stats
        avg_stats = {
            'reward': np.mean(episode_rewards),
            'loss': np.mean([s['loss'] for s in episode_stats]),
            'value_estimate': np.mean([s['value_estimate'] for s in episode_stats]),
            'advantage': np.mean([s['advantage'] for s in episode_stats]),
            'avg_policy_loss': np.mean([s['policy_loss'] for s in episode_stats]),
            'avg_value_loss': np.mean([s['value_loss'] for s in episode_stats]),
            'total_distance': np.mean(episode_distances),
            'num_changes': np.mean(episode_changes),
            'selected_vertiports': new_selected_vertiports  # For best config tracking
        }
        
        # Log to metrics tracker
        if self.metrics_tracker:
            #                                   episode, episode_rewards, stats,     design_step_data
            self.metrics_tracker.log_episode(episode_num, episode_rewards, avg_stats, episode_stats)
        
        return episode_rewards, avg_stats
    
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
    
    uam_simulator.set_airspace_vp_design()
    
    if test_mode:
        print('Vertiport Design problem in test mode')
        uam_simulator.airspace.make_regions_dict_vp_des_test_mode(map_centeroid_to_region_center=2*(32_000_000**0.5), region_center_to_vp=32_000_000**0.5)
    else:
        uam_simulator.airspace.make_regions_dict_vp_des('commercial', num_regions=num_regions)

    # Initialize graph builder
    graph_builder = VertiportGraphBuilder(
        airspace=uam_simulator.airspace,
        node_feature_dim=8, # x,y, 1,1,1,1,region_id
        edge_feature_dim=7,
        connectivity_type='full' # option1: inter_intra
    )
    num_regions = len(uam_simulator.airspace.regions_dict)
    
    HIDDEN_DIM = 16
    SHARED_NO_ACTION = False

    # Initialize model
    rl_model = StationSelectionGNNRL(
        node_features=8,
        edge_features=7,
        hidden_dim=HIDDEN_DIM,
        num_regions=num_regions,
        num_gnn_layers=1,
        shared_no_action=SHARED_NO_ACTION
    )
    
    # ===== NEW: Initialize metrics tracker =====
    metrics_tracker = MetricsTracker(
        hidden_dim=HIDDEN_DIM,
        shared_no_action=SHARED_NO_ACTION,
        save_dir='./training_results',
        use_tensorboard=False  # Set to False to disable TensorBoard
    )
    
    # Initialize trainer with metrics tracker
    trainer = A2CTrainer(
        model=rl_model,
        graph_builder=graph_builder,
        lr=1e-4,
        gamma=0.99,
        value_coef=0.5,
        entropy_coef=0.01,
        metrics_tracker=metrics_tracker  # Pass the tracker
    )
    
    # Training loop
    num_episodes = 500
    print("\n" + "="*70)
    print("STARTING TRAINING WITH COMPREHENSIVE VISUALIZATION")
    print("="*70)
    print(f"Results will be saved to: {metrics_tracker.save_dir}")
    if metrics_tracker.use_tensorboard:
        print(f"TensorBoard: Run 'tensorboard --logdir={metrics_tracker.save_dir}'")
    print("="*70 + "\n")
    
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode} ===")
        
        rewards, stats = trainer.train_episode(
            simulator=uam_simulator,
            simulator_steps=3,
            num_design_steps=50,
            episode_num=episode
        )
        
        print(f"Episode {episode} Summary:")
        print(f"  Avg Reward: {stats['reward']:.3f}")
        print(f"  Avg Distance: {stats['total_distance']:.2f}")
        print(f"  Value Estimate: {stats['value_estimate']:.3f}")
        print(f"  Advantage: {stats['advantage']:.3f}")
        print(f"  Loss: {stats['loss']:.3f}")
    
    print(f'\nTRAINING COMPLETE: vertiport list: {uam_simulator.airspace.vertiport_list}')
    
    print("\n=== Training Complete ===")
    
    # ===== NEW: Save all plots and print summary =====
    print("\nGenerating final visualizations...")
    metrics_tracker.save_all_plots()
    metrics_tracker.print_summary()
    metrics_tracker.close()
    
    # Inference
    rl_model.eval()
    
    # Build graph with CURRENT state
    x, edge_index, edge_attr = graph_builder.build_graph(
        selected_vertiports=trainer.current_selected_vertiports,
        metrics=trainer.current_metrics
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
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS SAVED!")
    print(f"Check the '{metrics_tracker.save_dir}' directory for:")
    print("  - Real-time training progress plots")
    print("  - Detailed metrics analysis")
    print("  - Comprehensive summary dashboard")
    if metrics_tracker.use_tensorboard:
        print("  - TensorBoard logs")
    print("="*70)
