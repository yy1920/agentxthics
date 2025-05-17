#!/usr/bin/env python3
"""
Run the Ethical Frameworks experiment.

This script provides a command-line interface for running simulations
with agents that adhere to different ethical frameworks.
"""
import os
import sys
import argparse
from agentxthics.research.run_ethical_frameworks import run_ethical_framework_experiment

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Ethical Frameworks Experiment')
    
    parser.add_argument(
        '--config',
        type=str,
        default='config_ethical_frameworks.json',
        help='Path to config file (default: config_ethical_frameworks.json)'
    )
    
    parser.add_argument(
        '--llm',
        type=str,
        choices=['mock', 'gpt', 'gemini'],
        default='mock',
        help='LLM provider to use (default: mock)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Custom output directory'
    )
    
    parser.add_argument(
        '--rounds',
        type=int,
        help='Override number of simulation rounds'
    )
    
    parser.add_argument(
        '--initial-amount',
        type=int,
        help='Override initial resource amount'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for ethical frameworks experiment."""
    args = parse_args()
    
    # Print header
    print("\n" + "=" * 80)
    print("AgentXthics Ethical Frameworks Experiment".center(80))
    print("=" * 80)
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        return 1
    
    print(f"Running experiment with config file: {args.config}")
    print(f"LLM provider: {args.llm}")
    
    if args.output:
        print(f"Custom output directory: {args.output}")
    
    # Custom parameter overrides
    custom_params = {}
    if args.rounds:
        print(f"Overriding simulation rounds: {args.rounds}")
        custom_params['num_rounds'] = args.rounds
    
    if args.initial_amount:
        print(f"Overriding initial resource amount: {args.initial_amount}")
        custom_params['initial_amount'] = args.initial_amount
    
    print("\nStarting simulation...")
    
    # Run the experiment
    resource = run_ethical_framework_experiment(
        config_file=args.config,
        llm_provider=args.llm,
        output_dir=args.output,
        custom_params=custom_params
    )
    
    if resource:
        print("\nExperiment completed successfully.")
        return 0
    else:
        print("\nExperiment failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
