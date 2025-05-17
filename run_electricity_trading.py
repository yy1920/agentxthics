#!/usr/bin/env python
"""
Electricity Trading Game Runner

This script runs the electricity trading game simulation with customizable parameters.
It allows agents representing electricity trading companies to interact in a market
and tests various hypotheses about their behavior.
"""
import os
import sys
import argparse
from agentxthics.scenarios.electricity_trading_game import ElectricityTradingGame

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the Electricity Trading Game')
    parser.add_argument('--config', type=str, default='config_electricity_trading.json',
                      help='Path to configuration file')
    parser.add_argument('--analyze-only', action='store_true',
                      help='Only analyze results without running a new simulation')
    parser.add_argument('--run-dir', type=str, default=None,
                      help='Run directory to analyze (used with --analyze-only)')
    return parser.parse_args()

def main():
    """Main entry point for the electricity trading game."""
    args = parse_arguments()
    
    # Create the game
    game = ElectricityTradingGame(config_path=args.config)
    
    if args.analyze_only:
        # Only analyze a previous run
        if args.run_dir:
            print(f"Analyzing results in {args.run_dir}")
            analysis = game.analyze_results(run_dir=args.run_dir)
        else:
            print("Analyzing most recent run results")
            analysis = game.analyze_results()
        
        # Print key findings to console
        if "hypotheses" in analysis:
            print("\nKey Findings:")
            for name, results in analysis["hypotheses"].items():
                if "conclusion" in results:
                    title = " ".join(word.capitalize() for word in name.split("_"))
                    print(f"- {title}: {results['conclusion']}")
    else:
        # Run a new simulation and analyze results
        print("Setting up electricity trading simulation")
        game.setup()
        
        print("Running simulation")
        game.run()
        
        print("Analyzing results")
        analysis = game.analyze_results()
        
        # Print summary to console
        if "summary" in analysis:
            summary = analysis["summary"]
            print("\nSimulation Summary:")
            print(f"Rounds: {summary.get('num_rounds', 0)}")
            print(f"Agents: {summary.get('num_agents', 0)}")
            print(f"Total Trades: {summary.get('total_trades', 0):.2f} units")
            print(f"Average Price: ${summary.get('average_price', 0):.2f}")
            print(f"Total Shortages: {summary.get('total_shortages', 0)}")
            
            # Show agent profits
            agent_profits = summary.get("agent_profits", {})
            if agent_profits:
                print("\nAgent Profits:")
                sorted_agents = sorted(agent_profits.items(), key=lambda x: x[1].get("profit", 0), reverse=True)
                for agent_id, data in sorted_agents:
                    print(f"  Agent {agent_id} ({data.get('personality', 'unknown')}): ${data.get('profit', 0):.2f}")

if __name__ == "__main__":
    main()
