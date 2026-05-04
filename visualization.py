"""
Visualization Module
====================

NOVEL CONTRIBUTIONS:
1. Comprehensive plotting suite for RL metrics
2. Policy visualization with arrows
3. State-value heatmaps with customization
4. Convergence analysis plots
5. Comparison plots for multiple algorithms

Provides publication-quality visualizations for academic presentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from gridworld_env import EnhancedGridWorld


class RLVisualizer:
    """
    Visualization toolkit for Reinforcement Learning results.
    
    Generates plots for:
    - Training metrics (rewards, success rate, etc.)
    - Policy visualization
    - Value function heatmaps
    - Convergence analysis
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = figsize
        
    def plot_training_metrics(
        self,
        stats: Dict,
        algorithm_name: str,
        save_path: Optional[str] = None
    ):
        """
        Plot comprehensive training metrics.
        
        Creates a 2x2 grid showing:
        1. Episode rewards
        2. Success rate (moving average)
        3. Episode lengths
        4. Policy stability (changes per episode)
        
        Args:
            stats: Statistics dictionary from training
            algorithm_name: Name of algorithm (for title)
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(f'{algorithm_name} - Training Metrics', fontsize=16, fontweight='bold')
        
        # 1. Episode Rewards
        ax = axes[0, 0]
        episodes = range(1, len(stats['episode_rewards']) + 1)
        ax.plot(episodes, stats['episode_rewards'], alpha=0.3, color='steelblue', label='Raw')
        
        # Moving average
        window = min(50, len(stats['episode_rewards']) // 10)
        if window > 1:
            ma_rewards = self._moving_average(stats['episode_rewards'], window)
            ax.plot(episodes[window-1:], ma_rewards, color='darkblue', linewidth=2, 
                   label=f'{window}-Episode MA')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title('Episode Rewards', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Success Rate
        ax = axes[0, 1]
        success_rate = np.array(stats['success_episodes']) * 100
        window = min(50, len(success_rate) // 10)
        if window > 1:
            ma_success = self._moving_average(success_rate, window)
            ax.plot(episodes[window-1:], ma_success, color='green', linewidth=2)
        else:
            ax.plot(episodes, success_rate, color='green', linewidth=2)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title(f'Success Rate ({window}-Episode Moving Average)', fontsize=14, fontweight='bold')
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        
        # 3. Episode Lengths
        ax = axes[1, 0]
        ax.plot(episodes, stats['episode_lengths'], alpha=0.3, color='coral', label='Raw')
        
        if window > 1:
            ma_lengths = self._moving_average(stats['episode_lengths'], window)
            ax.plot(episodes[window-1:], ma_lengths, color='darkred', linewidth=2,
                   label=f'{window}-Episode MA')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Steps per Episode', fontsize=12)
        ax.set_title('Episode Lengths', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Policy Stability (Changes)
        ax = axes[1, 1]
        ax.plot(episodes, stats['policy_changes'], alpha=0.5, color='purple', label='Raw')
        
        if window > 1:
            ma_changes = self._moving_average(stats['policy_changes'], window)
            ax.plot(episodes[window-1:], ma_changes, color='darkviolet', linewidth=2,
                   label=f'{window}-Episode MA')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Policy Changes', fontsize=12)
        ax.set_title('Policy Stability (Lower = More Stable)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved training metrics to {save_path}")
        
        return fig
    
    def plot_policy_arrows(
        self,
        env: EnhancedGridWorld,
        policy_grid: np.ndarray,
        value_function: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        title: str = "Learned Policy"
    ):
        """
        Visualize policy as arrows on the grid.
        
        NOVELTY: Clear visual representation of learned policy with state annotations.
        
        Args:
            env: GridWorld environment
            policy_grid: Grid of action indices
            value_function: Optional value function for background coloring
            save_path: Path to save figure
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        grid_size = env.grid_size
        
        # Background: value function heatmap (if provided)
        if value_function is not None:
            im = ax.imshow(value_function, cmap='YlOrRd', alpha=0.6, interpolation='nearest')
            plt.colorbar(im, ax=ax, label='State Value')
        else:
            # Just a light background
            ax.imshow(np.zeros((grid_size, grid_size)), cmap='gray', alpha=0.1)
        
        # Draw grid
        for i in range(grid_size + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5)
            ax.axvline(i - 0.5, color='black', linewidth=0.5)
        
        # Arrow directions
        arrow_dict = {
            env.UP: (0, -0.35),
            env.DOWN: (0, 0.35),
            env.LEFT: (-0.35, 0),
            env.RIGHT: (0.35, 0)
        }
        
        # Draw policy arrows and state markers
        for i in range(grid_size):
            for j in range(grid_size):
                state = (i, j)
                state_type = env.get_state_type(state)
                
                # Mark special states
                if state_type == 'start':
                    circle = plt.Circle((j, i), 0.3, color='blue', alpha=0.7, zorder=5)
                    ax.add_patch(circle)
                    ax.text(j, i, 'S', ha='center', va='center', fontsize=14, 
                           fontweight='bold', color='white', zorder=6)
                
                elif state_type == 'goal':
                    circle = plt.Circle((j, i), 0.3, color='green', alpha=0.7, zorder=5)
                    ax.add_patch(circle)
                    ax.text(j, i, 'G', ha='center', va='center', fontsize=14,
                           fontweight='bold', color='white', zorder=6)
                
                elif state_type == 'penalty':
                    circle = plt.Circle((j, i), 0.3, color='red', alpha=0.7, zorder=5)
                    ax.add_patch(circle)
                    ax.text(j, i, 'T', ha='center', va='center', fontsize=14,
                           fontweight='bold', color='white', zorder=6)
                
                elif state_type == 'obstacle':
                    rect = plt.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8, 
                                        color='black', alpha=0.8, zorder=5)
                    ax.add_patch(rect)
                
                # Draw policy arrow for non-obstacle states
                elif policy_grid[i, j] >= 0:
                    action = int(policy_grid[i, j])
                    dx, dy = arrow_dict[action]
                    ax.arrow(j, i, dx, dy, head_width=0.15, head_length=0.1,
                            fc='darkblue', ec='darkblue', linewidth=2, zorder=4)
        
        # Legend
        legend_elements = [
            mpatches.Patch(color='blue', label='Start'),
            mpatches.Patch(color='green', label='Goal'),
            mpatches.Patch(color='red', label='Penalty/Trap'),
            mpatches.Patch(color='black', label='Obstacle'),
            mpatches.FancyArrow(0, 0, 0.1, 0, color='darkblue', label='Policy Action')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(grid_size - 0.5, -0.5)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved policy visualization to {save_path}")
        
        return fig
    
    def plot_value_heatmap(
        self,
        env: EnhancedGridWorld,
        value_function: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "State-Value Function"
    ):
        """
        Plot state-value function as a heatmap.
        
        Args:
            env: GridWorld environment
            value_function: Value function grid
            save_path: Path to save figure
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(value_function, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={'label': 'Value'},
                   ax=ax)
        
        # Mark special states
        grid_size = env.grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                state = (i, j)
                state_type = env.get_state_type(state)
                
                if state_type == 'start':
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                              edgecolor='blue', linewidth=3))
                    ax.text(j + 0.5, i + 0.15, 'START', ha='center', va='top',
                           fontsize=8, fontweight='bold', color='blue')
                
                elif state_type == 'goal':
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                              edgecolor='green', linewidth=3))
                    ax.text(j + 0.5, i + 0.15, 'GOAL', ha='center', va='top',
                           fontsize=8, fontweight='bold', color='green')
                
                elif state_type == 'penalty':
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                              edgecolor='red', linewidth=3))
                    ax.text(j + 0.5, i + 0.15, 'TRAP', ha='center', va='top',
                           fontsize=8, fontweight='bold', color='red')
        
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved value heatmap to {save_path}")
        
        return fig
    
    def plot_algorithm_comparison(
        self,
        stats_dict: Dict[str, Dict],
        metric: str = 'episode_rewards',
        save_path: Optional[str] = None
    ):
        """
        Compare multiple algorithms on a single metric.
        
        Args:
            stats_dict: Dictionary mapping algorithm names to their stats
            metric: Metric to plot ('episode_rewards', 'success_episodes', etc.)
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['steelblue', 'coral', 'green', 'purple', 'orange']
        
        for idx, (name, stats) in enumerate(stats_dict.items()):
            episodes = range(1, len(stats[metric]) + 1)
            data = stats[metric]
            
            # Convert success to percentage if needed
            if metric == 'success_episodes':
                data = np.array(data) * 100
            
            # Plot with moving average
            window = min(50, len(data) // 10)
            if window > 1:
                ma_data = self._moving_average(data, window)
                ax.plot(episodes[window-1:], ma_data, 
                       label=name, color=colors[idx % len(colors)], linewidth=2)
            else:
                ax.plot(episodes, data, label=name, 
                       color=colors[idx % len(colors)], linewidth=2)
        
        metric_labels = {
            'episode_rewards': ('Episode', 'Total Reward', 'Episode Rewards Comparison'),
            'success_episodes': ('Episode', 'Success Rate (%)', 'Success Rate Comparison'),
            'episode_lengths': ('Episode', 'Steps per Episode', 'Episode Length Comparison'),
            'policy_changes': ('Episode', 'Policy Changes', 'Policy Stability Comparison')
        }
        
        xlabel, ylabel, title = metric_labels.get(metric, ('Episode', metric, f'{metric} Comparison'))
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved comparison plot to {save_path}")
        
        return fig
    
    @staticmethod
    def _moving_average(data: List[float], window: int) -> np.ndarray:
        """Calculate moving average."""
        return np.convolve(data, np.ones(window)/window, mode='valid')


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Visualization Module")
    print("=" * 60)
    
    # Create dummy data for testing
    dummy_stats = {
        'episode_rewards': np.random.randn(500).cumsum() / 10,
        'episode_lengths': np.random.randint(10, 50, 500),
        'success_episodes': (np.random.random(500) > 0.5).astype(int),
        'policy_changes': np.maximum(0, 20 - np.arange(500) // 25)
    }
    
    vis = RLVisualizer()
    
    # Test training metrics plot
    print("\nGenerating training metrics plot...")
    vis.plot_training_metrics(dummy_stats, "Test Algorithm")
    plt.show()
    
    print("✓ Visualization test complete!")
