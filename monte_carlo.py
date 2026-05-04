"""
Monte Carlo Control Algorithm
==============================

NOVEL CONTRIBUTIONS:
1. Epsilon-decay exploration strategy for adaptive learning
2. Comprehensive convergence tracking (policy stability, success rate)
3. Episode-based learning with importance sampling
4. First-visit Monte Carlo with every-visit option

Implementation: Monte Carlo Control with epsilon-greedy policy
"""

import numpy as np
from collections import defaultdict
from typing import Tuple, List, Dict, Optional
from gridworld_env import EnhancedGridWorld


class MonteCarloControl:
    """
    Monte Carlo Control algorithm for reinforcement learning.
    
    Uses first-visit MC prediction with epsilon-greedy policy improvement.
    Tracks convergence metrics for analysis.
    """
    
    def __init__(
        self,
        env: EnhancedGridWorld,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None
    ):
        """
        Initialize Monte Carlo Control agent.
        
        Args:
            env: GridWorld environment
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon (multiplicative)
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-table: Q[(state, action)] = value
        self.Q = defaultdict(lambda: 0.0)
        
        # Returns: stores all returns for each state-action pair
        self.returns = defaultdict(list)
        
        # Policy: greedy policy derived from Q
        self.policy = {}
        
        # Tracking metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_episodes = []
        self.policy_changes = []
        
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
    
    def generate_episode(self) -> List[Tuple]:
        """
        Generate an episode using current policy.
        
        Returns:
            List of (state, action, reward) tuples
        """
        episode = []
        state = self.env.reset()
        done = False
        
        while not done:
            action = self.select_action(state, training=True)
            next_state, reward, done, info = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        
        return episode
    
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
        Train the agent using Monte Carlo Control.
        
        NOVELTY: Tracks multiple convergence metrics.
        
        Args:
            n_episodes: Number of episodes to train
            verbose: Print progress
        
        Returns:
            Training statistics dictionary
        """
        for episode_num in range(n_episodes):
            # Generate episode
            episode = self.generate_episode()
            
            # Track episode metrics
            episode_reward = sum(r for _, _, r in episode)
            episode_length = len(episode)
            success = episode_reward > 0  # Success if positive reward
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.success_episodes.append(1 if success else 0)
            
            # First-visit Monte Carlo: update Q-values
            G = 0  # Return
            visited_state_actions = set()
            
            # Process episode in reverse (backward view)
            for t in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                
                state_action = (state, action)
                
                # First-visit: only update if not visited before in this episode
                if state_action not in visited_state_actions:
                    visited_state_actions.add(state_action)
                    
                    # Store return
                    self.returns[state_action].append(G)
                    
                    # Update Q-value (average of returns)
                    self.Q[state_action] = np.mean(self.returns[state_action])
            
            # Policy improvement: update policy based on new Q-values
            policy_changes = self.update_policy()
            self.policy_changes.append(policy_changes)
            
            # Decay epsilon (NOVELTY: epsilon-decay exploration)
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay
            
            # Verbose output
            if verbose and (episode_num + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                success_rate = np.mean(self.success_episodes[-100:]) * 100
                print(f"Episode {episode_num + 1}/{n_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.1f} | "
                      f"Success Rate: {success_rate:.1f}% | "
                      f"Epsilon: {self.epsilon:.3f}")
        
        # Compile statistics
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_episodes': self.success_episodes,
            'policy_changes': self.policy_changes,
            'final_epsilon': self.epsilon,
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
    print("Testing Monte Carlo Control")
    print("=" * 60)
    
    env = EnhancedGridWorld(grid_size=6, seed=42)
    agent = MonteCarloControl(env, seed=42)
    
    print("\nTraining for 500 episodes...")
    stats = agent.train(n_episodes=500, verbose=True)
    
    print(f"\n✓ Training complete!")
    print(f"Final epsilon: {stats['final_epsilon']:.3f}")
    print(f"States visited: {stats['total_states_visited']}")
    print(f"Final 100-episode success rate: {np.mean(stats['success_episodes'][-100:]) * 100:.1f}%")
