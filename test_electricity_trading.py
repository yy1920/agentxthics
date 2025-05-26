"""
Test script for electricity trading game.

This script runs a simulation with the modified code and verifies that
all expected log files are created.
"""
import os
import json
import sys
from datetime import datetime

from agentxthics.scenarios.electricity_trading_game import ElectricityTradingGame

def main():
    """Run a test simulation and verify logs."""
    print("Starting test electricity trading simulation...")
    
    # Create a simple config for quick testing
    config = {
        "output_dir": "logs/electricity_trading",
        "market": {
            "initial_price": 40,
            "price_volatility": 0.4,
            "num_rounds": 5  # Short simulation for testing
        },
        "llm": {
            "type": "openai",  # Always use OpenAI for better agent interactions
            "model": "gpt-4"
        },
        "agents": [
            {
                "id": "A",
                "personality": "cooperative",
                "profit_bias": 0.3,
                "generation_capacity": 100,
                "storage_capacity": 50,
                "demand_profile": "steady"
            },
            {
                "id": "B",
                "personality": "competitive",
                "profit_bias": 0.7,
                "generation_capacity": 80,
                "storage_capacity": 40,
                "demand_profile": "variable"
            },
            {
                "id": "C",
                "personality": "adaptive",
                "profit_bias": 0.5,
                "generation_capacity": 120,
                "storage_capacity": 60,
                "demand_profile": "peak"
            }
        ],
        "communication": {
            "enable_bilateral_negotiation": True,
            "enable_public_announcements": True,
            "max_contracts_per_turn": 2,
            "force_contract_proposals": True,
            "min_contracts_per_round": 1
        }
    }
    
    # Create temp config file
    with open("test_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Run simulation
    game = ElectricityTradingGame("test_config.json")
    game.setup().run()
    
    # Find the latest run directory
    output_dir = config["output_dir"]
    dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("run_")]
    latest_run = os.path.join(output_dir, sorted(dirs)[-1])
    
    print(f"Checking logs in {latest_run}...")
    
    # Check for expected log files
    expected_files = [
        "message_log.json",
        "decision_log.json",
        "contract_log.json", 
        "state_log.json",
        "shortage_log.json",
        "trade_log.json",
        "summary.json"
    ]
    
    missing_files = []
    for file in expected_files:
        path = os.path.join(latest_run, file)
        if not os.path.exists(path):
            missing_files.append(file)
        else:
            # Check if file has content
            with open(path, "r") as f:
                content = json.load(f)
                print(f"- {file}: {len(content)} entries")
    
    if missing_files:
        print(f"ERROR: Missing expected log files: {', '.join(missing_files)}")
        return 1
    else:
        print("SUCCESS: All expected log files were created")
        return 0

if __name__ == "__main__":
    sys.exit(main())
