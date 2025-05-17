import os
import sys
import time
from enhanced_simulation import run_simulation

# Create output directories
for scenario in ['cooperative_scenario', 'competitive_scenario', 'mixed_scenario', 'tragedy_scenario']:
    os.makedirs(scenario, exist_ok=True)

# Run Cooperative Scenario
print("\n=== Running Cooperative Scenario ===")
coop_config = {
    'output_dir': 'cooperative_scenario',
    'resource': {
        'initial_amount': 50,
        'conserve_amount': 5,
        'consume_amount': 10,
        'default_renewal': 15,
        'bonus_renewal': 20,
        'num_rounds': 10
    },
    'agents': [
        {'id': 'A1', 'cooperation_bias': 0.9, 'personality': 'cooperative'},
        {'id': 'A2', 'cooperation_bias': 0.9, 'personality': 'cooperative'},
        {'id': 'A3', 'cooperation_bias': 0.9, 'personality': 'cooperative'}
    ]
}
run_simulation(coop_config)

# Run Competitive Scenario
print("\n=== Running Competitive Scenario ===")
comp_config = {
    'output_dir': 'competitive_scenario',
    'resource': {
        'initial_amount': 50,
        'conserve_amount': 5,
        'consume_amount': 10,
        'default_renewal': 15,
        'bonus_renewal': 20,
        'num_rounds': 10
    },
    'agents': [
        {'id': 'A1', 'cooperation_bias': 0.3, 'personality': 'competitive'},
        {'id': 'A2', 'cooperation_bias': 0.3, 'personality': 'competitive'},
        {'id': 'A3', 'cooperation_bias': 0.3, 'personality': 'competitive'}
    ]
}
run_simulation(comp_config)

# Run Mixed Scenario
print("\n=== Running Mixed Agents Scenario ===")
mixed_config = {
    'output_dir': 'mixed_scenario',
    'resource': {
        'initial_amount': 50,
        'conserve_amount': 5,
        'consume_amount': 10,
        'default_renewal': 15,
        'bonus_renewal': 20,
        'num_rounds': 10
    },
    'agents': [
        {'id': 'A1', 'cooperation_bias': 0.9, 'personality': 'cooperative'},
        {'id': 'A2', 'cooperation_bias': 0.5, 'personality': 'adaptive'},
        {'id': 'A3', 'cooperation_bias': 0.3, 'personality': 'competitive'}
    ]
}
run_simulation(mixed_config)

# Run Tragedy of the Commons Scenario
print("\n=== Running Tragedy of the Commons Scenario ===")
tragedy_config = {
    'output_dir': 'tragedy_scenario',
    'resource': {
        'initial_amount': 30,
        'conserve_amount': 5,
        'consume_amount': 10,
        'default_renewal': 10,
        'bonus_renewal': 15,
        'num_rounds': 10
    },
    'agents': [
        {'id': 'A1', 'cooperation_bias': 0.4, 'rationality': 0.5},
        {'id': 'A2', 'cooperation_bias': 0.4, 'rationality': 0.5},
        {'id': 'A3', 'cooperation_bias': 0.4, 'rationality': 0.5}
    ]
}
run_simulation(tragedy_config)

# Generate comparison visualization
print("\n=== Generating Scenario Comparisons ===")
import matplotlib.pyplot as plt
import json
import numpy as np

# Create comparison plot
plt.figure(figsize=(10, 6))

# Load data from each scenario
scenarios = ['cooperative_scenario', 'competitive_scenario', 'mixed_scenario', 'tragedy_scenario']
scenario_names = ['Cooperative', 'Competitive', 'Mixed Agents', 'Tragedy of the Commons']
colors = ['green', 'red', 'blue', 'orange']

for i, (scenario, name, color) in enumerate(zip(scenarios, scenario_names, colors)):
    state_log_path = os.path.join(scenario, 'state_log.json')
    if os.path.exists(state_log_path):
        with open(state_log_path, 'r') as f:
            data = json.load(f)
        
        rounds = [state['round'] for state in data]
        amounts = [state.get('amount', 50) for state in data]
        
        plt.plot(rounds, amounts, 'o-', label=name, color=color)

plt.title('Resource Pool Comparison Across Scenarios')
plt.xlabel('Round')
plt.ylabel('Pool Amount')
plt.legend()
plt.grid(True)

# Save comparison plot
plt.tight_layout()
plt.savefig('scenario_comparison.png')
print("Scenario comparison visualization saved as scenario_comparison.png")

print("\nAll scenarios completed successfully!")
