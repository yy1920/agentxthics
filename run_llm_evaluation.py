#!/usr/bin/env python
"""
Run the LLM-as-a-Judge evaluation on an electricity trading simulation run.

This script provides a command-line interface for running the LLM-as-a-Judge
evaluation on a specific simulation run directory.

Example usage:
    python run_llm_evaluation.py --run_dir logs/electricity_trading/run_20250517-143241 --model_provider gemini
"""
import os
import sys
import argparse

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment variables loaded from .env file")
except ImportError:
    print("python-dotenv package not installed. Environment variables may not be properly loaded.")

from agentxthics.evaluation.llm_judge import run_llm_judge_evaluation


def main():
    """Run the LLM-as-a-Judge evaluation on a simulation run."""
    parser = argparse.ArgumentParser(description="Run LLM-as-a-Judge evaluation on an electricity trading simulation")
    
    parser.add_argument(
        "--run_dir", 
        type=str, 
        required=True,
        help="Path to the simulation run directory"
    )
    parser.add_argument(
        "--model_provider", 
        type=str, 
        default="gemini",
        choices=["openai", "gemini"],
        help="LLM provider to use (openai or gemini)"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default=None,
        help="Specific model to use (default: gemini-1.5-pro for Gemini, gpt-4-turbo for OpenAI)"
    )
    parser.add_argument(
        "--hypotheses",
        type=str,
        nargs="*",
        help="Specific hypotheses to evaluate (default: all)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more details during evaluation"
    )
    
    args = parser.parse_args()
    
    # Validate run directory
    if not os.path.exists(args.run_dir):
        print(f"Error: Run directory '{args.run_dir}' does not exist.")
        sys.exit(1)
    
    # Set default model name based on provider if not specified
    model_name = args.model_name
    if model_name is None:
        model_name = "gemini-1.5-pro" if args.model_provider == "gemini" else "gpt-4-turbo"
    
    print(f"Running LLM-as-a-Judge evaluation on: {args.run_dir}")
    print(f"Using {args.model_provider} model: {model_name}")
    
    # Run the evaluation
    try:
        results = run_llm_judge_evaluation(args.run_dir, args.model_provider, model_name)
        
        # Print the file paths to the results
        print(f"\nResults saved to:")
        print(f"  - {os.path.join(args.run_dir, 'llm_evaluation.json')}")
        print(f"  - {os.path.join(args.run_dir, 'llm_evaluation_report.txt')}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
