#!/usr/bin/env python3
"""
Test script for ethical frameworks experiment.
This is a simplified version that just verifies the basic functionality works.
"""
import os
import shutil
import json
import datetime
from agentxthics.research.run_ethical_frameworks import run_ethical_framework_experiment

def main():
    """Run a test of the ethical frameworks experiment."""
    # Create a test output directory
    test_dir = "game_logs/test_ethical_frameworks"
    os.makedirs(test_dir, exist_ok=True)
    
    # Run the experiment with mock LLM
    print("\n=== Testing Ethical Frameworks Experiment ===")
    print(f"Test output directory: {test_dir}")
    
    # Load the config file
    config_file = "config_ethical_frameworks.json"
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' not found.")
        return 1
    
    # Run the experiment with specific output directory
    resource = run_ethical_framework_experiment(
        config_file=config_file,
        llm_provider="mock",
        output_dir=test_dir
    )
    
    # Check if the experiment created any files
    if resource is not None:
        # Look for the run directory
        run_dirs = [d for d in os.listdir(test_dir) if d.startswith("run_")]
        if run_dirs:
            latest_run = sorted(run_dirs)[-1]
            run_path = os.path.join(test_dir, latest_run)
            
            # Print list of files created
            print(f"\nFiles created in {run_path}:")
            for root, _, files in os.walk(run_path):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), run_path)
                    print(f"  {rel_path}")
            
            # Print summary if it exists
            summary_file = os.path.join(run_path, 'simulation_summary.txt')
            if os.path.exists(summary_file):
                print("\nSimulation Summary:")
                with open(summary_file, 'r') as f:
                    print(f.read())
            
            print("\nTest completed successfully!")
            return 0
        else:
            print("\nError: No run directories found in the output directory.")
            return 1
    else:
        print("\nError: Experiment failed to run.")
        return 1

if __name__ == "__main__":
    main()
