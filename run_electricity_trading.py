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
import logging
from datetime import datetime
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
    parser.add_argument('--log-file', type=str, default=None,
                      help='Path to save real-time logs (default: simulation_TIMESTAMP.log)')
    return parser.parse_args()

def setup_logging(log_file=None):
    """Set up logging to file and console."""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"simulation_{timestamp}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Create file handler which logs even debug messages
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler with the same log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging to {os.path.abspath(log_file)}")
    return log_file

def main():
    """Main entry point for the electricity trading game."""
    args = parse_arguments()
    
    # Set up logging
    log_file = setup_logging(args.log_file)
    
    # Create the game
    game = ElectricityTradingGame(config_path=args.config)
    
    if args.analyze_only:
        # Only analyze a previous run
        if args.run_dir:
            logging.info(f"Analyzing results in {args.run_dir}")
            analysis = game.analyze_results(run_dir=args.run_dir)
        else:
            logging.info("Analyzing most recent run results")
            analysis = game.analyze_results()
        
        # Print key findings to console
        if "hypotheses" in analysis:
            logging.info("\nKey Findings:")
            for name, results in analysis["hypotheses"].items():
                if "conclusion" in results:
                    title = " ".join(word.capitalize() for word in name.split("_"))
                    logging.info(f"- {title}: {results['conclusion']}")
    else:
        # Run a new simulation and analyze results
        logging.info("Setting up electricity trading simulation")
        game.setup()
        
        logging.info("Running simulation")
        game.run()
        
        logging.info("Analyzing results")
        analysis = game.analyze_results()
        
        # Print summary to console
        if "summary" in analysis:
            summary = analysis["summary"]
            logging.info("\nSimulation Summary:")
            logging.info(f"Rounds: {summary.get('num_rounds', 0)}")
            logging.info(f"Agents: {summary.get('num_agents', 0)}")
            logging.info(f"Total Trades: {summary.get('total_trades', 0):.2f} units")
            logging.info(f"Average Price: ${summary.get('average_price', 0):.2f}")
            logging.info(f"Total Shortages: {summary.get('total_shortages', 0)}")
            
            # Show agent profits
            agent_profits = summary.get("agent_profits", {})
            if agent_profits:
                logging.info("\nAgent Profits:")
                sorted_agents = sorted(agent_profits.items(), key=lambda x: x[1].get("profit", 0), reverse=True)
                for agent_id, data in sorted_agents:
                    logging.info(f"  Agent {agent_id} ({data.get('personality', 'unknown')}): ${data.get('profit', 0):.2f}")
        
        logging.info(f"Full logs saved to {log_file}")

if __name__ == "__main__":
    main()
