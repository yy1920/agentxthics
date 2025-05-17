#!/usr/bin/env python3
"""
Run debug simulation with enhanced logging to diagnose the timeout issues.
This script runs the simulation with the example configuration and outputs detailed logs.
"""
import os
import sys
import json
from debug_simulation import run_debug_simulation

def main():
    """Run debug simulation and analyze results."""
    # Create logs directory
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set custom output directory
    custom_dir = os.path.join(logs_dir, "debug_run")
    os.makedirs(custom_dir, exist_ok=True)
    
    print(f"Starting debug simulation with example config")
    print(f"Logs will be saved to: {custom_dir}")
    
    # First try with example config
    config_file = os.path.join(os.path.dirname(__file__), "example_config.json")
    
    # Load config and modify it
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            print(f"Loaded example config with {len(config.get('agents', []))} agents")
            
            # Update config for debugging
            config['output_dir'] = custom_dir
            
            # Run debug simulation
            results = run_debug_simulation(config=config)
            
            # Analyze timeout events
            if 'logs' in results and 'timeout_log' in results['logs']:
                timeout_log_path = results['logs']['timeout_log']
                
                with open(timeout_log_path, 'r') as f:
                    timeout_data = json.load(f)
                
                if timeout_data:
                    print("\nTimeout events detected:")
                    for event in timeout_data:
                        round_num = event.get('round', 'unknown')
                        pending = event.get('pending_agents', [])
                        resource_state = event.get('resource_state', 'unknown')
                        print(f"- Round {round_num}: Agents {pending} timed out when resource was {resource_state}")
                    
                    # Generate summary report
                    summary_path = os.path.join(custom_dir, "timeout_summary.txt")
                    with open(summary_path, 'w') as f:
                        f.write("Timeout Events Analysis\n")
                        f.write("=====================\n\n")
                        
                        # Group by agent
                        agent_timeouts = {}
                        for event in timeout_data:
                            for agent in event.get('pending_agents', []):
                                if agent not in agent_timeouts:
                                    agent_timeouts[agent] = []
                                agent_timeouts[agent].append(event.get('round', 'unknown'))
                        
                        f.write("Timeouts by agent:\n")
                        for agent, rounds in agent_timeouts.items():
                            f.write(f"Agent {agent}: {len(rounds)} timeouts in rounds {rounds}\n")
                        
                        f.write("\nPossible causes:\n")
                        f.write("1. Resource shortage - Agent requesting more resources than available\n")
                        f.write("2. Decision logic errors - Agents stuck in decision making\n")
                        f.write("3. Communication issues - Agents waiting for messages that never arrive\n")
                        f.write("4. Concurrency bugs - Race conditions in the simpy environment\n")
                else:
                    print("\nNo timeout events detected in this run")
            
            print("\nDebug simulation completed")
            print(f"Check {custom_dir} for detailed logs")
    else:
        print(f"Error: Could not find {config_file}")
        # Run with default configuration
        run_debug_simulation()

if __name__ == "__main__":
    main()
