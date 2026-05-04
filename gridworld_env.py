"""
Enhanced GridWorld Environment for Reinforcement Learning
==========================================================

NOVEL CONTRIBUTIONS:
1. Dynamic obstacles that relocate every episode
2. Multiple goal states with different reward magnitudes
3. Configurable penalty/trap states
4. Stochastic action execution (slip probability)
5. Flexible grid sizing (6x6, 8x8, or custom)

Author: RL Mini Project
Course: Machine Learning / AI
"""

import numpy as np
import random
from typing import Tuple, List, Dict, Optional


class EnhancedGridWorld:
    """
    An enhanced GridWorld environment with dynamic elements.
    
    Features:
    - Configurable grid size
    - Dynamic obstacles that change position each episode
    - Multiple goals with varying rewards
    - Penalty states (traps)
    - Stochastic actions (slip probability)
    """
    
    # Action space
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    ACTIONS = [UP, DOWN, LEFT, RIGHT]
    ACTION_NAMES = {UP: '↑', DOWN: '↓', LEFT: '←', RIGHT: '→'}
    
    def __init__(
        self,
        grid_size: int = 6,
        n_dynamic_obstacles: int = 3,
        n_goals: int = 2,
        n_penalty_states: int = 2,
        slip_probability: float = 0.2,
        goal_rewards: Optional[List[float]] = None,
        penalty_values: Optional[List[float]] = None,
        step_penalty: float = -0.01,
        seed: Optional[int] = None
    ):
        """
        Initialize the Enhanced GridWorld.
        
        Args:
            grid_size: Size of the square grid (e.g., 6 for 6x6)
            n_dynamic_obstacles: Number of obstacles that move each episode
            n_goals: Number of goal states
            n_penalty_states: Number of penalty/trap states
            slip_probability: Probability of action failing and moving randomly
            goal_rewards: List of rewards for each goal (default: [10, 5])
            penalty_values: List of penalties for traps (default: [-5, -3])
            step_penalty: Small penalty for each step to encourage efficiency
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.grid_size = grid_size
        self.n_dynamic_obstacles = n_dynamic_obstacles
        self.n_goals = n_goals
        self.n_penalty_states = n_penalty_states
        self.slip_probability = slip_probability
        self.step_penalty = step_penalty
        
        # Set default rewards if not provided
        self.goal_rewards = goal_rewards if goal_rewards else [10.0, 5.0]
        self.penalty_values = penalty_values if penalty_values else [-5.0, -3.0]
        
        # Ensure we have enough rewards/penalties
        while len(self.goal_rewards) < n_goals:
            self.goal_rewards.append(self.goal_rewards[-1] * 0.7)
        while len(self.penalty_values) < n_penalty_states:
            self.penalty_values.append(self.penalty_values[-1] * 0.7)
        
        # State space
        self.n_states = grid_size * grid_size
        self.n_actions = len(self.ACTIONS)
        
        # Initialize state positions
        self.start_state = (0, 0)  # Top-left corner
        self.current_state = self.start_state
        
        # These will be set during reset()
        self.goal_states = []
        self.penalty_states = []
        self.obstacle_positions = []
        
        # Episode tracking
        self.episode_steps = 0
        self.max_steps = grid_size * grid_size * 2  # Prevent infinite episodes
        
    def _get_available_positions(self, exclude: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Get all grid positions excluding specified ones."""
        all_positions = [(i, j) for i in range(self.grid_size) 
                        for j in range(self.grid_size)]
        return [pos for pos in all_positions if pos not in exclude]
    
    def reset(self) -> Tuple[int, int]:
        """
        Reset the environment for a new episode.
        
        NOVELTY: Dynamic obstacles are repositioned each episode.
        
        Returns:
            Starting state (row, col)
        """
        self.current_state = self.start_state
        self.episode_steps = 0
        
        # Reserve positions
        reserved = [self.start_state]
        
        # Place goal states (fixed positions for learning stability)
        # But with some randomness for novelty
        if not hasattr(self, '_fixed_goals'):
            available = self._get_available_positions(reserved)
            self.goal_states = random.sample(available, self.n_goals)
            self._fixed_goals = self.goal_states.copy()
        else:
            self.goal_states = self._fixed_goals.copy()
        
        reserved.extend(self.goal_states)
        
        # Place penalty states (fixed positions)
        if not hasattr(self, '_fixed_penalties'):
            available = self._get_available_positions(reserved)
            self.penalty_states = random.sample(available, self.n_penalty_states)
            self._fixed_penalties = self.penalty_states.copy()
        else:
            self.penalty_states = self._fixed_penalties.copy()
        
        reserved.extend(self.penalty_states)
        
        # NOVELTY: Place dynamic obstacles (change every episode)
        available = self._get_available_positions(reserved)
        if len(available) >= self.n_dynamic_obstacles:
            self.obstacle_positions = random.sample(available, self.n_dynamic_obstacles)
        else:
            self.obstacle_positions = available
        
        return self.current_state
    
    def _is_valid_position(self, state: Tuple[int, int]) -> bool:
        """Check if a position is valid (within grid and not an obstacle)."""
        row, col = state
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            return False
        if state in self.obstacle_positions:
            return False
        return True
    
    def _get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Calculate next state based on action."""
        row, col = state
        
        if action == self.UP:
            next_state = (row - 1, col)
        elif action == self.DOWN:
            next_state = (row + 1, col)
        elif action == self.LEFT:
            next_state = (row, col - 1)
        elif action == self.RIGHT:
            next_state = (row, col + 1)
        else:
            next_state = state
        
        # If next state is invalid, stay in current state
        if not self._is_valid_position(next_state):
            return state
        
        return next_state
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """
        Execute action in the environment.
        
        NOVELTY: Stochastic action execution with slip probability.
        
        Args:
            action: Action to take (UP, DOWN, LEFT, RIGHT)
        
        Returns:
            next_state: Resulting state
            reward: Reward received
            done: Whether episode is finished
            info: Additional information
        """
        self.episode_steps += 1
        
        # NOVELTY: Stochastic action (slip probability)
        if np.random.random() < self.slip_probability:
            # Action fails, move in random direction
            actual_action = np.random.choice(self.ACTIONS)
        else:
            actual_action = action
        
        # Get next state
        next_state = self._get_next_state(self.current_state, actual_action)
        
        # Calculate reward
        reward = self.step_penalty  # Small penalty for each step
        done = False
        info = {'slip_occurred': actual_action != action}
        
        # Check if reached goal
        if next_state in self.goal_states:
            goal_idx = self.goal_states.index(next_state)
            reward = self.goal_rewards[goal_idx]
            done = True
            info['termination_reason'] = 'goal_reached'
        
        # Check if hit penalty state
        elif next_state in self.penalty_states:
            penalty_idx = self.penalty_states.index(next_state)
            reward = self.penalty_values[penalty_idx]
            done = True
            info['termination_reason'] = 'penalty_state'
        
        # Check if max steps reached
        elif self.episode_steps >= self.max_steps:
            done = True
            reward = -1.0  # Penalty for timeout
            info['termination_reason'] = 'max_steps'
        
        self.current_state = next_state
        
        return next_state, reward, done, info
    
    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (row, col) state to single index."""
        row, col = state
        return row * self.grid_size + col
    
    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert single index to (row, col) state."""
        row = index // self.grid_size
        col = index % self.grid_size
        return (row, col)
    
    def get_state_type(self, state: Tuple[int, int]) -> str:
        """Return the type of state for visualization."""
        if state == self.start_state:
            return 'start'
        elif state in self.goal_states:
            return 'goal'
        elif state in self.penalty_states:
            return 'penalty'
        elif state in self.obstacle_positions:
            return 'obstacle'
        else:
            return 'empty'
    
    def render(self) -> str:
        """
        Render the current state of the grid.
        
        Returns:
            String representation of the grid
        """
        grid = []
        for i in range(self.grid_size):
            row = []
            for j in range(self.grid_size):
                state = (i, j)
                if state == self.current_state:
                    row.append(' A ')  # Agent
                elif state == self.start_state:
                    row.append(' S ')  # Start
                elif state in self.goal_states:
                    row.append(' G ')  # Goal
                elif state in self.penalty_states:
                    row.append(' T ')  # Trap
                elif state in self.obstacle_positions:
                    row.append(' X ')  # Obstacle
                else:
                    row.append(' . ')  # Empty
            grid.append('|' + '|'.join(row) + '|')
        
        return '\n'.join(grid)


if __name__ == "__main__":
    # Test the environment
    print("=" * 60)
    print("Testing Enhanced GridWorld Environment")
    print("=" * 60)
    
    env = EnhancedGridWorld(grid_size=6, n_dynamic_obstacles=3, seed=42)
    
    print("\nEpisode 1:")
    state = env.reset()
    print(env.render())
    print(f"\nGoal states: {env.goal_states}")
    print(f"Penalty states: {env.penalty_states}")
    print(f"Obstacles: {env.obstacle_positions}")
    
    print("\nEpisode 2 (obstacles should move):")
    state = env.reset()
    print(env.render())
    print(f"New obstacles: {env.obstacle_positions}")
    
    print("\n✓ Environment test complete!")
