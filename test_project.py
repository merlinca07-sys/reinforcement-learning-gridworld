"""
Quick Test Script - Verify All Components
==========================================

This script runs a quick verification of all project components
without running the full 2000-episode experiment.

Usage: python test_project.py
"""

import numpy as np
from gridworld_env import EnhancedGridWorld
from monte_carlo import MonteCarloControl
from sarsa import SARSA


def test_environment():
    """Test environment creation and basic functionality."""
    print("=" * 60)
    print("TEST 1: Environment")
    print("=" * 60)
    
    env = EnhancedGridWorld(grid_size=6, seed=42)
    state = env.reset()
    
    print("✓ Environment created successfully")
    print(f"  Grid size: {env.grid_size}x{env.grid_size}")
    print(f"  State space: {env.n_states}")
    print(f"  Action space: {env.n_actions}")
    
    # Test episode
    done = False
    steps = 0
    while not done and steps < 100:
        action = np.random.choice(env.ACTIONS)
        state, reward, done, info = env.step(action)
        steps += 1
    
    print(f"✓ Episode completed in {steps} steps")
    print()


def test_monte_carlo():
    """Test Monte Carlo Control."""
    print("=" * 60)
    print("TEST 2: Monte Carlo Control")
    print("=" * 60)
    
    env = EnhancedGridWorld(grid_size=6, seed=42)
    agent = MonteCarloControl(env, seed=42)
    
    print("✓ Agent created successfully")
    
    # Quick training
    stats = agent.train(n_episodes=100, verbose=False)
    
    print(f"✓ Training completed")
    print(f"  Episodes: 100")
    print(f"  Final success rate: {np.mean(stats['success_episodes'][-20:]) * 100:.1f}%")
    print(f"  States visited: {stats['total_states_visited']}")
    print()


def test_sarsa():
    """Test SARSA."""
    print("=" * 60)
    print("TEST 3: SARSA")
    print("=" * 60)
    
    env = EnhancedGridWorld(grid_size=6, seed=42)
    agent = SARSA(env, seed=42)
    
    print("✓ Agent created successfully")
    
    # Quick training
    stats = agent.train(n_episodes=100, verbose=False)
    
    print(f"✓ Training completed")
    print(f"  Episodes: 100")
    print(f"  Final success rate: {np.mean(stats['success_episodes'][-20:]) * 100:.1f}%")
    print(f"  States visited: {stats['total_states_visited']}")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ENHANCED GRIDWORLD RL PROJECT - QUICK TEST")
    print("=" * 60)
    print()
    
    try:
        test_environment()
        test_monte_carlo()
        test_sarsa()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("Ready to run full experiment:")
        print("  python main.py")
        print()
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
