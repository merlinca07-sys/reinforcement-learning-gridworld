"""
Main Execution Script - Enhanced GridWorld RL Project
======================================================

NOVEL MINI PROJECT CONTRIBUTIONS:
1. Dynamic obstacle GridWorld environment
2. Multiple goals with varying rewards
3. Penalty/trap states for risk-reward tradeoffs
4. Stochastic action execution
5. Epsilon-decay exploration strategies
6. Comprehensive convergence tracking
7. Policy stability metrics
8. Comparative analysis of MC vs SARSA

This script orchestrates the complete experimental pipeline:
- Environment setup
- Algorithm training (Monte Carlo & SARSA)
- Performance evaluation
- Visualization generation
- Results comparison

Author: RL Mini Project
Usage: python main.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict

from gridworld_env import EnhancedGridWorld
from monte_carlo import MonteCarloControl
from sarsa import SARSA
from visualization import RLVisualizer


class RLExperiment:
    """
    Orchestrates the complete RL experiment pipeline.
    
    Handles:
    - Environment creation
    - Agent training
    - Results collection
    - Visualization generation
    - Comparative analysis
    """
    
    def __init__(
        self,
        grid_size: int = 6,
        n_episodes: int = 2000,
        seed: int = 42,
        output_dir: str = "results"
    ):
        """
        Initialize experiment.
        
        Args:
            grid_size: Size of GridWorld (6x6 or 8x8 recommended)
            n_episodes: Number of training episodes
            seed: Random seed for reproducibility
            output_dir: Directory to save results
        """
        self.grid_size = grid_size
        self.n_episodes = n_episodes
        self.seed = seed
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = RLVisualizer()
        
        print("=" * 70)
        print("ENHANCED GRIDWORLD RL PROJECT")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Grid Size: {grid_size}x{grid_size}")
        print(f"  Episodes: {n_episodes}")
        print(f"  Seed: {seed}")
        print(f"  Output Dir: {output_dir}")
        print("\n" + "=" * 70)
    
    def create_environment(self) -> EnhancedGridWorld:
        """
        Create the enhanced GridWorld environment.
        
        NOVELTY: Dynamic obstacles, multiple goals, penalty states.
        """
        print("\n📍 Creating Enhanced GridWorld Environment...")
        
        env = EnhancedGridWorld(
            grid_size=self.grid_size,
            n_dynamic_obstacles=3,  # NOVELTY: obstacles move each episode
            n_goals=2,  # NOVELTY: multiple goals with different rewards
            n_penalty_states=2,  # NOVELTY: penalty/trap states
            slip_probability=0.2,  # NOVELTY: stochastic actions
            goal_rewards=[10.0, 5.0],  # Different rewards for goals
            penalty_values=[-5.0, -3.0],  # Different penalties for traps
            step_penalty=-0.01,  # Small penalty to encourage efficiency
            seed=self.seed
        )
        
        # Reset to initialize positions
        env.reset()
        
        print(f"✓ Environment created:")
        print(f"  - State space: {env.n_states} states")
        print(f"  - Action space: {env.n_actions} actions")
        print(f"  - Goal states: {env.goal_states} (rewards: {env.goal_rewards[:env.n_goals]})")
        print(f"  - Penalty states: {env.penalty_states} (penalties: {env.penalty_values[:env.n_penalty_states]})")
        print(f"  - Dynamic obstacles: {env.n_dynamic_obstacles} (positions change each episode)")
        print(f"  - Slip probability: {env.slip_probability}")
        
        print(f"\nCurrent Grid Configuration:")
        print(env.render())
        
        return env
    
    def train_monte_carlo(self, env: EnhancedGridWorld) -> Dict:
        """
        Train Monte Carlo Control agent.
        
        Returns:
            Training statistics
        """
        print("\n" + "=" * 70)
        print("🎲 TRAINING: Monte Carlo Control")
        print("=" * 70)
        
        agent = MonteCarloControl(
            env=env,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,  # NOVELTY: epsilon-decay
            seed=self.seed
        )
        
        print(f"\nHyperparameters:")
        print(f"  - Gamma (discount): {agent.gamma}")
        print(f"  - Epsilon: {agent.epsilon_start} → {agent.epsilon_end} (decay: {agent.epsilon_decay})")
        print(f"\nTraining for {self.n_episodes} episodes...\n")
        
        stats = agent.train(n_episodes=self.n_episodes, verbose=True)
        
        # Final statistics
        final_100_success = np.mean(stats['success_episodes'][-100:]) * 100
        final_100_reward = np.mean(stats['episode_rewards'][-100:])
        final_100_length = np.mean(stats['episode_lengths'][-100:])
        
        print(f"\n✓ Training Complete!")
        print(f"  - Final epsilon: {stats['final_epsilon']:.4f}")
        print(f"  - States visited: {stats['total_states_visited']}")
        print(f"  - Last 100 episodes:")
        print(f"    • Success rate: {final_100_success:.1f}%")
        print(f"    • Avg reward: {final_100_reward:.2f}")
        print(f"    • Avg length: {final_100_length:.1f} steps")
        
        # Add agent and environment to stats for later use
        stats['agent'] = agent
        stats['env'] = env
        
        return stats
    
    def train_sarsa(self, env: EnhancedGridWorld) -> Dict:
        """
        Train SARSA agent.
        
        Returns:
            Training statistics
        """
        print("\n" + "=" * 70)
        print("🎯 TRAINING: SARSA")
        print("=" * 70)
        
        agent = SARSA(
            env=env,
            gamma=0.99,
            alpha=0.1,
            alpha_decay=0.9999,  # NOVELTY: adaptive learning rate
            alpha_min=0.01,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,  # NOVELTY: epsilon-decay
            seed=self.seed
        )
        
        print(f"\nHyperparameters:")
        print(f"  - Gamma (discount): {agent.gamma}")
        print(f"  - Alpha (learning rate): {agent.alpha_start} → {agent.alpha_min} (decay: {agent.alpha_decay})")
        print(f"  - Epsilon: {agent.epsilon_start} → {agent.epsilon_end} (decay: {agent.epsilon_decay})")
        print(f"\nTraining for {self.n_episodes} episodes...\n")
        
        stats = agent.train(n_episodes=self.n_episodes, verbose=True)
        
        # Final statistics
        final_100_success = np.mean(stats['success_episodes'][-100:]) * 100
        final_100_reward = np.mean(stats['episode_rewards'][-100:])
        final_100_length = np.mean(stats['episode_lengths'][-100:])
        
        print(f"\n✓ Training Complete!")
        print(f"  - Final epsilon: {stats['final_epsilon']:.4f}")
        print(f"  - Final alpha: {stats['final_alpha']:.6f}")
        print(f"  - States visited: {stats['total_states_visited']}")
        print(f"  - Last 100 episodes:")
        print(f"    • Success rate: {final_100_success:.1f}%")
        print(f"    • Avg reward: {final_100_reward:.2f}")
        print(f"    • Avg length: {final_100_length:.1f} steps")
        
        # Add agent and environment to stats for later use
        stats['agent'] = agent
        stats['env'] = env
        
        return stats
    
    def generate_visualizations(
        self,
        mc_stats: Dict,
        sarsa_stats: Dict
    ):
        """
        Generate all visualizations for the project.
        
        Creates:
        1. Training metrics for each algorithm
        2. Policy visualizations
        3. Value function heatmaps
        4. Comparison plots
        """
        print("\n" + "=" * 70)
        print("📊 GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        # 1. Monte Carlo Training Metrics
        print("\n1. Monte Carlo training metrics...")
        self.visualizer.plot_training_metrics(
            mc_stats,
            "Monte Carlo Control",
            save_path=f"{self.output_dir}/mc_training_metrics.png"
        )
        plt.close()
        
        # 2. SARSA Training Metrics
        print("2. SARSA training metrics...")
        self.visualizer.plot_training_metrics(
            sarsa_stats,
            "SARSA",
            save_path=f"{self.output_dir}/sarsa_training_metrics.png"
        )
        plt.close()
        
        # 3. Monte Carlo Policy Visualization
        print("3. Monte Carlo policy visualization...")
        mc_agent = mc_stats['agent']
        mc_env = mc_stats['env']
        mc_policy = mc_agent.get_policy_grid()
        mc_values = mc_agent.get_value_function()
        
        self.visualizer.plot_policy_arrows(
            mc_env,
            mc_policy,
            mc_values,
            save_path=f"{self.output_dir}/mc_policy_arrows.png",
            title="Monte Carlo - Learned Policy"
        )
        plt.close()
        
        # 4. SARSA Policy Visualization
        print("4. SARSA policy visualization...")
        sarsa_agent = sarsa_stats['agent']
        sarsa_env = sarsa_stats['env']
        sarsa_policy = sarsa_agent.get_policy_grid()
        sarsa_values = sarsa_agent.get_value_function()
        
        self.visualizer.plot_policy_arrows(
            sarsa_env,
            sarsa_policy,
            sarsa_values,
            save_path=f"{self.output_dir}/sarsa_policy_arrows.png",
            title="SARSA - Learned Policy"
        )
        plt.close()
        
        # 5. Monte Carlo Value Heatmap
        print("5. Monte Carlo value heatmap...")
        self.visualizer.plot_value_heatmap(
            mc_env,
            mc_values,
            save_path=f"{self.output_dir}/mc_value_heatmap.png",
            title="Monte Carlo - State-Value Function"
        )
        plt.close()
        
        # 6. SARSA Value Heatmap
        print("6. SARSA value heatmap...")
        self.visualizer.plot_value_heatmap(
            sarsa_env,
            sarsa_values,
            save_path=f"{self.output_dir}/sarsa_value_heatmap.png",
            title="SARSA - State-Value Function"
        )
        plt.close()
        
        # 7-10. Comparison plots
        print("7. Reward comparison...")
        self.visualizer.plot_algorithm_comparison(
            {'Monte Carlo': mc_stats, 'SARSA': sarsa_stats},
            metric='episode_rewards',
            save_path=f"{self.output_dir}/comparison_rewards.png"
        )
        plt.close()
        
        print("8. Success rate comparison...")
        self.visualizer.plot_algorithm_comparison(
            {'Monte Carlo': mc_stats, 'SARSA': sarsa_stats},
            metric='success_episodes',
            save_path=f"{self.output_dir}/comparison_success_rate.png"
        )
        plt.close()
        
        print("9. Episode length comparison...")
        self.visualizer.plot_algorithm_comparison(
            {'Monte Carlo': mc_stats, 'SARSA': sarsa_stats},
            metric='episode_lengths',
            save_path=f"{self.output_dir}/comparison_lengths.png"
        )
        plt.close()
        
        print("10. Policy stability comparison...")
        self.visualizer.plot_algorithm_comparison(
            {'Monte Carlo': mc_stats, 'SARSA': sarsa_stats},
            metric='policy_changes',
            save_path=f"{self.output_dir}/comparison_policy_stability.png"
        )
        plt.close()
        
        print(f"\n✓ All visualizations saved to '{self.output_dir}/' directory")
    
    def run_complete_experiment(self):
        """
        Run the complete experimental pipeline.
        """
        # Step 1: Create environment
        env = self.create_environment()
        
        # Step 2: Train Monte Carlo
        mc_stats = self.train_monte_carlo(env)
        
        # Step 3: Reset environment and train SARSA
        env_sarsa = self.create_environment()  # New instance for fair comparison
        sarsa_stats = self.train_sarsa(env_sarsa)
        
        # Step 4: Generate all visualizations
        self.generate_visualizations(mc_stats, sarsa_stats)
        
        # Step 5: Final summary
        self.print_final_summary(mc_stats, sarsa_stats)
    
    def print_final_summary(self, mc_stats: Dict, sarsa_stats: Dict):
        """
        Print final summary comparing both algorithms.
        """
        print("\n" + "=" * 70)
        print("📈 FINAL SUMMARY & COMPARISON")
        print("=" * 70)
        
        mc_final_success = np.mean(mc_stats['success_episodes'][-100:]) * 100
        sarsa_final_success = np.mean(sarsa_stats['success_episodes'][-100:]) * 100
        
        mc_final_reward = np.mean(mc_stats['episode_rewards'][-100:])
        sarsa_final_reward = np.mean(sarsa_stats['episode_rewards'][-100:])
        
        mc_final_length = np.mean(mc_stats['episode_lengths'][-100:])
        sarsa_final_length = np.mean(sarsa_stats['episode_lengths'][-100:])
        
        print(f"\nFinal Performance (Last 100 Episodes):")
        print(f"\n  Monte Carlo Control:")
        print(f"    - Success Rate: {mc_final_success:.1f}%")
        print(f"    - Avg Reward: {mc_final_reward:.2f}")
        print(f"    - Avg Steps: {mc_final_length:.1f}")
        
        print(f"\n  SARSA:")
        print(f"    - Success Rate: {sarsa_final_success:.1f}%")
        print(f"    - Avg Reward: {sarsa_final_reward:.2f}")
        print(f"    - Avg Steps: {sarsa_final_length:.1f}")
        
        print(f"\nConvergence Analysis:")
        print(f"  - MC Policy Changes (final 100): {np.sum(mc_stats['policy_changes'][-100:])}")
        print(f"  - SARSA Policy Changes (final 100): {np.sum(sarsa_stats['policy_changes'][-100:])}")
        
        print(f"\nNovel Contributions Demonstrated:")
        print(f"  ✓ Dynamic obstacles (changed every episode)")
        print(f"  ✓ Multiple goals with different rewards")
        print(f"  ✓ Penalty states for risk-reward tradeoffs")
        print(f"  ✓ Stochastic action execution")
        print(f"  ✓ Epsilon-decay exploration strategy")
        print(f"  ✓ Comprehensive convergence tracking")
        print(f"  ✓ Policy stability metrics")
        print(f"  ✓ Comparative algorithm analysis")
        
        print("\n" + "=" * 70)
        print("✅ EXPERIMENT COMPLETE!")
        print(f"All results saved to: {self.output_dir}/")
        print("=" * 70 + "\n")


def main():
    """
    Main execution function.
    
    Customize experiment parameters here.
    """
    # Experiment configuration
    GRID_SIZE = 6  # Try 6x6 or 8x8
    N_EPISODES = 2000  # Number of training episodes
    SEED = 42  # For reproducibility
    
    # Create and run experiment
    experiment = RLExperiment(
        grid_size=GRID_SIZE,
        n_episodes=N_EPISODES,
        seed=SEED,
        output_dir="results"
    )
    
    experiment.run_complete_experiment()


if __name__ == "__main__":
    main()
