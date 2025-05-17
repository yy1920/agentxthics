"""
Main entry point for running AgentXthics research.
This module provides a unified interface for running various research scenarios.
"""
import os
import argparse
import json
from dotenv import load_dotenv

from agentxthics.research.scenarios import (
    run_simulation,
    run_asymmetric_scenario, 
    run_vulnerable_scenario,
    run_ethical_framework_comparison
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run AgentXthics Research Scenarios')
    
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['asymmetric', 'vulnerable', 'ethical', 'custom'],
        default='asymmetric',
        help='Research scenario to run'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration JSON file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='logs/research',
        help='Directory to save research logs and results'
    )
    
    parser.add_argument(
        '--rounds',
        type=int,
        default=10,
        help='Number of simulation rounds'
    )
    
    parser.add_argument(
        '--agents',
        type=int,
        default=4,
        help='Number of agents (for custom scenario only)'
    )
    
    parser.add_argument(
        '--initial-amount',
        type=int,
        default=50,
        help='Initial resource amount'
    )
    
    parser.add_argument(
        '--enable-ethics',
        action='store_true',
        help='Enable ethical frameworks'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for research scenarios."""
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Parse command line arguments
    args = parse_args()
    
    # Set base output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Set ethical frameworks flag
    if args.enable_ethics:
        os.environ['ENABLE_ETHICAL_FRAMEWORKS'] = 'true'
    
    print(f"\n=== AgentXthics Research Simulation ===")
    print(f"Scenario: {args.scenario}")
    print(f"Output directory: {output_dir}")
    
    if args.scenario == 'asymmetric':
        # Run asymmetric information scenario
        scenario_dir = os.path.join(output_dir, 'asymmetric_scenario')
        resource = run_asymmetric_scenario(scenario_dir)
        
        print(f"\nAsymmetric information scenario completed.")
        print(f"Results saved to {scenario_dir}")
        
    elif args.scenario == 'vulnerable':
        # Run vulnerable populations scenario
        scenario_dir = os.path.join(output_dir, 'vulnerable_scenario')
        resource = run_vulnerable_scenario(scenario_dir)
        
        print(f"\nVulnerable populations scenario completed.")
        print(f"Results saved to {scenario_dir}")
        
    elif args.scenario == 'ethical':
        # Run ethical framework comparison
        scenario_dir = os.path.join(output_dir, 'ethical_frameworks')
        results = run_ethical_framework_comparison(scenario_dir)
        
        print(f"\nEthical framework comparison completed.")
        print(f"Results saved to {scenario_dir}")
        
    elif args.scenario == 'custom':
        # Run custom scenario with provided configuration
        if args.config and os.path.exists(args.config):
            # Load configuration from file
            with open(args.config, 'r') as f:
                try:
                    config = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error: Could not parse configuration file {args.config}")
                    return
        else:
            # Create configuration from command line arguments
            config = {
                'output_dir': os.path.join(output_dir, 'custom_scenario'),
                'resource': {
                    'initial_amount': args.initial_amount,
                    'num_rounds': args.rounds
                }
            }
            
            # Create default agents if none are specified
            if not config.get('agents'):
                config['agents'] = []
                for i in range(args.agents):
                    if i < args.agents // 3:
                        personality = 'cooperative'
                        cooperation_bias = 0.8
                    elif i < 2 * args.agents // 3:
                        personality = 'adaptive'
                        cooperation_bias = 0.5
                    else:
                        personality = 'competitive'
                        cooperation_bias = 0.3
                    
                    config['agents'].append({
                        'id': f'A{i+1}',
                        'personality': personality,
                        'cooperation_bias': cooperation_bias
                    })
        
        # Run simulation with custom configuration
        resource = run_simulation(config)
        
        print(f"\nCustom scenario completed.")
        print(f"Results saved to {config['output_dir']}")
    
    print("\nSimulation complete!")

if __name__ == "__main__":
    main()
