#!/usr/bin/env python3
"""Direct SUMO traffic simulator launcher - no descriptions, just GUI."""

import os
import sys
from pathlib import Path

# Add parent directory to path so we can import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set SUMO_HOME before importing
from scripts.common import load_config, ensure_sumo_home

ensure_sumo_home()

import time
import yaml
import torch
import numpy as np
from src.env.sumo_env import EnvConfig, SumoMDPEnv, VNWeights
from src.dqn.agent import DQNAgent, AgentConfig
from src.baseline import FixedTimeController, FixedTimeConfig


def run_simulation(mode: str = "dqn", steps: int = 3600) -> None:
    """Run SUMO simulation with GUI visualization."""
    config = load_config()
    
    # Enable GUI mode for realistic 2D visualization
    gui_mode = True
    
    # Extract config values with safe defaults
    sumo_cfg = config.get('sumo', {})
    vn_cfg = config.get('vn_weights', {})
    
    # Create environment with GUI visualization
    env_config = EnvConfig(
        # Use GUI config if available, otherwise use standard config
        sumocfg_path=sumo_cfg.get('sumocfg_path', 'data/scenarios/hn_sample/config_gui.sumocfg' if gui_mode else 'data/scenarios/hn_sample/config.sumocfg'),
        tls_id=sumo_cfg.get('tls_id', '0'),
        phases=sumo_cfg.get('phases', [25, 5, 25, 5]),
        action_duration=sumo_cfg.get('action_duration', 1),
        max_steps=steps,
        gui=gui_mode,  # Enable SUMO-GUI with enhanced visualization
        vn_weights=VNWeights(
            motorcycle=vn_cfg.get('motorcycle', 0.6),
            car=vn_cfg.get('car', 0.3),
            bus=vn_cfg.get('bus', 0.05),
            truck=vn_cfg.get('truck', 0.05),
        ),
    )
    
    print("\n🚦 Starting SUMO Traffic Simulator...")
    print(f"Mode: {mode}")
    print(f"Steps: {steps}")
    print(f"Config: {env_config.sumocfg_path}")
    
    env = SumoMDPEnv(env_config)
    state = env.reset()
    state_dim = state.shape[0] if isinstance(state, np.ndarray) else len(state)
    action_dim = len(sumo_cfg.get('phases', [25, 5, 25, 5]))
    
    # Load model if DQN mode and model exists
    agent: DQNAgent | None = None
    if mode == "dqn":
        model_path = Path("outputs/dqn_vn_tls.pt")
        if model_path.exists():
            try:
                agent_cfg = AgentConfig(state_dim=state_dim, action_dim=action_dim)
                agent = DQNAgent(agent_cfg)
                checkpoint = torch.load(model_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    agent.q.load_state_dict(checkpoint['model_state_dict'])
                else:
                    agent.q.load_state_dict(checkpoint)
                agent.q.eval()
            except Exception:
                agent = None
    
    # Initialize controller for fixed-time mode
    controller: FixedTimeController | None = None
    if mode == "fixed":
        controller = FixedTimeController(
            FixedTimeConfig(green_duration=25, yellow_duration=5)
        )
    
    # Run simulation
    print("\n⏱️  Running simulation...\n")
    total_reward = 0
    step = 0
    for step in range(steps):
        if mode == "demo":
            # Random action
            action = np.random.randint(action_dim)
        elif mode == "dqn" and agent is not None:
            # DQN action
            action = agent.act(state, eps=0.0)
        elif controller is not None:
            # Fixed-time controller
            action = controller.get_action()
        else:
            # Fallback to random
            action = np.random.randint(action_dim)
        
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        # Small delay to allow GUI to render (50ms per step = realistic speed)
        time.sleep(0.05)
        
        # Progress update every 500 steps
        if (step + 1) % 500 == 0:
            print(f"Step {step + 1}/{steps} | Reward: {reward:.2f} | Total: {total_reward:.2f}")
        
        if done:
            break
    
    print(f"\n✅ Simulation completed!")
    print(f"Total steps: {step + 1}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward: {total_reward/(step+1):.2f}")
    
    env.close()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dqn"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
    run_simulation(mode, steps)
