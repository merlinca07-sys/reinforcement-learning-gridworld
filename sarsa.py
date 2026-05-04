"""
SARSA Algorithm
===============

NOVEL CONTRIBUTIONS:
1. On-policy TD learning with epsilon-decay
2. Adaptive learning rate scheduling
3. Convergence tracking and policy stability metrics
4. Comparison-ready implementation alongside Monte Carlo

Implementation: SARSA (State-Action-Reward-State-Action)
"""

import numpy as np
from collections import defaultdict
from typing import Tuple, List, Dict, Optional
from gridworld_env import EnhancedGridWorld


class SARSA:
    """
    SARSA (On-policy TD Control) algorithm for reinforcement learning.
    
    Updates Q-values using the actual next action taken (on-policy).
    Includes epsilon-decay and learning rate scheduling.
    """
    
    def __init__(
        self,
        env: EnhancedGridWorld,
        gamma: float = 0.99,
        alpha: float = 0.1,
        alpha_decay: float = 0.9999,
        alpha_min: float = 0.01,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None
    ):
        """
        Initialize SARSA agent.
        
        Args:
            env: GridWorld environment
            gamma: Discount factor
            alpha: Learning rate
            alpha_decay: Decay rate for learning rate
            alpha_min: Minimum learning rate
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_start = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-table: Q[(state, action)] = value
        self.Q = defaultdict(lambda: 0.0)
        
        # Policy: greedy policy derived from Q
        self.policy = {}
        
        # Tracking metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_episodes = []
        self.policy_changes = []
        self.td_errors = []  # Track TD errors for convergence analysis
        
    def select_action(self, state: Tuple[int, int], training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: If True, use epsilon-greedy; if False, use greedy
        
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(self.env.ACTIONS)
        else:
            # Exploit: best known action
            q_values = [self.Q[(state, a)] for a in self.env.ACTIONS]
            max_q = max(q_values)
            # Break ties randomly
            best_actions = [a for a in self.env.ACTIONS if self.Q[(state, a)] == max_q]
            return np.random.choice(best_actions)
    
    def update_policy(self) -> int:
        """
        Update policy based on current Q-values.
        
        Returns:
            Number of states where policy changed
        """
        old_policy = self.policy.copy()
        changes = 0
        
        # Get all states that have been visited
        states = set(state for (state, _) in self.Q.keys())
        
        for state in states:
            # Find best action
            q_values = [self.Q[(state, a)] for a in self.env.ACTIONS]
            best_action = self.env.ACTIONS[np.argmax(q_values)]
            
            if state in old_policy and old_policy[state] != best_action:
                changes += 1
            
            self.policy[state] = best_action
        
        return changes
    
    def train(self, n_episodes: int, verbose: bool = True) -> Dict:
        """
        Train the agent using SARSA.
        
        NOVELTY: On-policy learning with adaptive learning rate and comprehensive tracking.
        
        Args:
            n_episodes: Number of episodes to train
            verbose: Print progress
        
        Returns:
            Training statistics dictionary
        """
        for episode_num in range(n_episodes):
            # Reset environment
            state = self.env.reset()
            action = self.select_action(state, training=True)
            
            episode_reward = 0
            episode_length = 0
            episode_td_errors = []
            done = False
            
            while not done:
                # Take action
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                
                # Select next action (on-policy: using current policy)
                next_action = self.select_action(next_state, training=True)
                
                # SARSA update: Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
                current_q = self.Q[(state, action)]
                next_q = self.Q[(next_state, next_action)] if not done else 0
                
                # TD error
                td_error = reward + self.gamma * next_q - current_q
                episode_td_errors.append(abs(td_error))
                
                # Update Q-value
                self.Q[(state, action)] += self.alpha * td_error
                
                # Move to next state-action pair
                state = next_state
                action = next_action
            
            # Track episode metrics
            success = episode_reward > 0
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.success_episodes.append(1 if success else 0)
            self.td_errors.append(np.mean(episode_td_errors) if episode_td_errors else 0)
            
            # Policy improvement
            policy_changes = self.update_policy()
            self.policy_changes.append(policy_changes)
            
            # Decay epsilon (NOVELTY: epsilon-decay exploration)
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay
            
            # Decay learning rate (NOVELTY: adaptive learning rate)
            if self.alpha > self.alpha_min:
                self.alpha *= self.alpha_decay
            
            # Verbose output
            if verbose and (episode_num + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                success_rate = np.mean(self.success_episodes[-100:]) * 100
                avg_td_error = np.mean(self.td_errors[-100:])
                print(f"Episode {episode_num + 1}/{n_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.1f} | "
                      f"Success Rate: {success_rate:.1f}% | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Alpha: {self.alpha:.4f} | "
                      f"TD Error: {avg_td_error:.4f}")
        
        # Compile statistics
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_episodes': self.success_episodes,
            'policy_changes': self.policy_changes,
            'td_errors': self.td_errors,
            'final_epsilon': self.epsilon,
            'final_alpha': self.alpha,
            'total_states_visited': len(set(s for (s, _) in self.Q.keys()))
        }
        
        return stats
    
    def get_value_function(self) -> np.ndarray:
        """
        Extract state-value function from Q-values.
        
        Returns:
            Value function as grid_size x grid_size array
        """
        V = np.zeros((self.env.grid_size, self.env.grid_size))
        
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                state = (i, j)
                # V(s) = max_a Q(s, a)
                q_values = [self.Q[(state, a)] for a in self.env.ACTIONS]
                V[i, j] = max(q_values) if any(q_values) else 0
        
        return V
    
    def get_policy_grid(self) -> np.ndarray:
        """
        Get policy as a grid of actions.
        
        Returns:
            Policy grid (grid_size x grid_size) with action indices
        """
        policy_grid = np.full((self.env.grid_size, self.env.grid_size), -1)
        
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                state = (i, j)
                if state in self.policy:
                    policy_grid[i, j] = self.policy[state]
                else:
                    # Default to best Q-value if policy not explicitly set
                    q_values = [self.Q[(state, a)] for a in self.env.ACTIONS]
                    if any(q != 0 for q in q_values):
                        policy_grid[i, j] = np.argmax(q_values)
        
        return policy_grid


if __name__ == "__main__":
    print("=" * 60)
    print("Testing SARSA")
    print("=" * 60)
    
    env = EnhancedGridWorld(grid_size=6, seed=42)
    agent = SARSA(env, seed=42)
    
    print("\nTraining for 500 episodes...")
    stats = agent.train(n_episodes=500, verbose=True)
    
    print(f"\n✓ Training complete!")
    print(f"Final epsilon: {stats['final_epsilon']:.3f}")
    print(f"Final alpha: {stats['final_alpha']:.4f}")
    print(f"States visited: {stats['total_states_visited']}")
    print(f"Final 100-episode success rate: {np.mean(stats['success_episodes'][-100:]) * 100:.1f}%")
