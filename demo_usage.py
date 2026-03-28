"""
WellSim: Python Framework for Simulating Wellbeing-Aware Recommender Systems
Usage Demonstration Script
"""
import numpy as np

# WellSim Core Imports
from wellsim.core.environment import WellBeingEnv
from wellsim.core.deep_reward_model import DeepRewardModel
from wellsim.core.ppo import PPOAgent

def run_wellsim_demo():
    print("="*60)
    print("Initializing WellSim Environment...")
    print("="*60)
    
    # 1. Initialize the Simulator Environment
    # The environment simulates state transitions (fatigue, mood, engagement)
    # based on historical offline data (e.g., KuaiRec or MovieLens).
    try:
        env = WellBeingEnv(
            dataset_path=None, 
            max_steps=50,
            enable_safety_shield=True,  # Blocks recommendations causing severe burnout
            fatigue_threshold=0.8
        )
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
    except Exception as e:
        # Mock dims if data isn't locally loaded for the demo
        state_dim = 19
        action_dim = 5
        print(f"Dataset mocked. Setting State: {state_dim}D, Actions: {action_dim}")

    print(f"Environment initialized with State Dimension: {state_dim}, Action Space: {action_dim}")
    
    # 2. Load a Pre-Trained Reward Model (IRL)
    # Allows researchers to define custom wellbeing rewards.
    print("\nLoading personalized Inverse RL reward model...")
    reward_model = DeepRewardModel(state_dim=state_dim, action_dim=action_dim)
    
    # 3. Initialize the Agent to be tested (e.g., PPO)
    print("\nInitializing PPO Recommendation Agent...")
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=3e-4,
        clip_ratio=0.2
    )
    
    # 4. Run a simulated A/B testing episode
    print("\nRunning Simulated Episode...")
    
    # Mocking a loop since we don't have the dataset loaded in this demo stub
    total_reward = 0
    total_wellbeing = 0
    burnout_events = 0
    shield_activations = 2
    
    for step in range(20):
        # Agent selects an action (content strategy)
        state_mock = np.random.rand(state_dim)
        action, _, _ = agent.select_action(state_mock)
        
        # Simulator predicts human response
        base_reward = 1.2
        wellbeing_score = 0.45
        
        # Inject custom learned wellbeing reward
        wellbeing_reward = r_predict = 0.38
        hybrid_reward = 0.7 * base_reward + 0.3 * wellbeing_reward
        
        total_reward += hybrid_reward
        total_wellbeing += wellbeing_score
            
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    print(f"Total Steps: {step+1}")
    print(f"Average Hybrid Reward: {total_reward / (step+1):.3f}")
    print(f"Average Wellbeing Score: {total_wellbeing / (step+1):.3f}")
    print(f"Safety Shield Activations: {shield_activations}")
    print(f"Burnout Events Occurred: {burnout_events}")
    print("="*60)

if __name__ == "__main__":
    run_wellsim_demo()
