#!/usr/bin/env python
"""
Example script demonstrating how to analyze electricity trading simulation results
using the LLM-as-a-Judge approach.

This script shows how to:
1. Load simulation logs from a specific run
2. Run the LLM-as-a-Judge evaluation on these logs
3. Visualize the evaluation results
4. Compare the LLM's assessment with quantitative metrics
"""
import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


from agentxthics.evaluation.llm_judge import LLMJudge, run_llm_judge_evaluation


def load_evaluation_results(run_dir):
    """Load existing LLM evaluation results if available."""
    eval_path = os.path.join(run_dir, "llm_evaluation.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            return json.load(f)
    return None


def visualize_hypothesis_evaluation(results, output_dir=None):
    """Create visualizations of the LLM judge's evaluation results."""
    # Create a directory for visualizations if it doesn't exist
    if output_dir is None:
        output_dir = "evaluation_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract hypothesis evaluations
    hypotheses = [h for h in results.keys() if h != "overall_summary"]
    
    # Create a confidence level visualization
    confidence_levels = []
    conclusions = []
    for hypothesis in hypotheses:
        confidence = results[hypothesis].get("confidence", "").lower()
        if confidence == "high":
            confidence_value = 3
        elif confidence == "medium":
            confidence_value = 2
        elif confidence == "low":
            confidence_value = 1
        else:
            confidence_value = 0
        
        confidence_levels.append(confidence_value)
        
        conclusion = results[hypothesis].get("conclusion", "").lower()
        if "support" in conclusion or "confirm" in conclusion or "true" in conclusion:
            conclusions.append("Supported")
        elif "refute" in conclusion or "reject" in conclusion or "false" in conclusion:
            conclusions.append("Refuted")
        else:
            conclusions.append("Inconclusive")
    
    # Create a bar chart of confidence levels
    plt.figure(figsize=(12, 6))
    bars = plt.bar(hypotheses, confidence_levels)
    
    # Color bars based on conclusion
    colors = {"Supported": "green", "Refuted": "red", "Inconclusive": "gray"}
    for i, conclusion in enumerate(conclusions):
        bars[i].set_color(colors[conclusion])
    
    plt.title("LLM Judge Hypothesis Evaluation")
    plt.xlabel("Hypothesis")
    plt.ylabel("Confidence Level (1=Low, 2=Medium, 3=High)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Add conclusion labels to each bar
    for i, (bar, conclusion) in enumerate(zip(bars, conclusions)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            conclusion,
            ha="center",
            color=colors[conclusion],
            weight="bold"
        )
    
    plt.savefig(os.path.join(output_dir, "hypothesis_evaluation.png"))
    print(f"Saved visualization to {os.path.join(output_dir, 'hypothesis_evaluation.png')}")
    
    # Create a pie chart of overall results
    supported = conclusions.count("Supported")
    refuted = conclusions.count("Refuted")
    inconclusive = conclusions.count("Inconclusive")
    
    plt.figure(figsize=(8, 8))
    plt.pie(
        [supported, refuted, inconclusive],
        labels=["Supported", "Refuted", "Inconclusive"],
        colors=["green", "red", "gray"],
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Overall Hypothesis Assessment")
    plt.savefig(os.path.join(output_dir, "overall_assessment.png"))
    print(f"Saved visualization to {os.path.join(output_dir, 'overall_assessment.png')}")


def analyze_evidence_patterns(results):
    """Analyze patterns in the evidence provided by the LLM judge."""
    # Extract all evidence statements
    all_evidence = []
    for hypothesis, data in results.items():
        if hypothesis == "overall_summary":
            continue
        
        for evidence in data.get("evidence", []):
            all_evidence.append({
                "hypothesis": hypothesis,
                "evidence": evidence,
                "supported": "support" in data.get("conclusion", "").lower() or "confirm" in data.get("conclusion", "").lower()
            })
    
    # Create a DataFrame for analysis
    df = pd.DataFrame(all_evidence)
    
    # Print some statistics about the evidence
    if not df.empty:
        print("\nEvidence Analysis:")
        print(f"Total evidence statements: {len(df)}")
        print(f"Average evidence statements per hypothesis: {len(df) / len(results) - 1:.1f}")
        print(f"Evidence supporting hypotheses: {df['supported'].sum()} ({df['supported'].mean() * 100:.1f}%)")
        
        # Look for common terms in evidence
        all_text = " ".join(df["evidence"].tolist()).lower()
        common_terms = ["contract", "price", "communication", "agent", "decision", "strategy", "pattern", "behavior"]
        print("\nCommon terms in evidence:")
        for term in common_terms:
            count = all_text.count(term)
            print(f"  - '{term}': {count} mentions")
    
    return df


def compare_with_quantitative_metrics(run_dir, results):
    """Compare the LLM judge's assessment with quantitative metrics from the simulation."""
    # Load summary data
    summary_path = os.path.join(run_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        
        print("\nComparison with Quantitative Metrics:")
        
        # Compare communication impact
        if "communication_impact" in results:
            comm_conclusion = results["communication_impact"].get("conclusion", "").lower()
            comm_supported = "support" in comm_conclusion or "confirm" in comm_conclusion
            
            # Get communication stats
            comm_count = summary.get("communication_count", 0)
            contract_count = summary.get("contract_count", 0)
            comm_ratio = comm_count / max(1, contract_count)
            
            print(f"Communication impact:")
            print(f"  - LLM assessment: {'Supported' if comm_supported else 'Not supported'}")
            print(f"  - Messages per contract: {comm_ratio:.2f}")
        
        # Compare profit vs stability
        if "profit_vs_stability" in results:
            profit_conclusion = results["profit_vs_stability"].get("conclusion", "").lower()
            profit_supported = "support" in profit_conclusion or "confirm" in profit_conclusion
            
            # Get shortage stats
            shortages = summary.get("total_shortages", 0)
            rounds = summary.get("num_rounds", 1)
            shortage_ratio = shortages / rounds
            
            print(f"Profit vs stability:")
            print(f"  - LLM assessment: {'Supported' if profit_supported else 'Not supported'}")
            print(f"  - Shortage ratio: {shortage_ratio:.2f} (shortages per round)")
            
            # Get profit variation
            agent_profits = summary.get("agent_profits", {})
            if agent_profits:
                profits = [data.get("profit", 0) for data in agent_profits.values()]
                profit_std = np.std(profits)
                profit_range = max(profits) - min(profits)
                
                print(f"  - Profit std deviation: ${profit_std:.2f}")
                print(f"  - Profit range: ${profit_range:.2f}")


def run_analysis(run_dir, regenerate=False, model_provider="gemini", model_name=None):
    """Run or load the LLM-as-a-Judge evaluation and analyze the results."""
    # Set default model name based on provider if not specified
    if model_name is None:
        model_name = "gemini-1.5-pro" if model_provider == "gemini" else "gpt-4-turbo"
    
    # Check if evaluation results already exist
    results = None
    if not regenerate:
        results = load_evaluation_results(run_dir)
    
    # Run the evaluation if needed
    if results is None:
        print(f"Running LLM-as-a-Judge evaluation on {run_dir}...")
        results = run_llm_judge_evaluation(run_dir, model_provider, model_name)
    else:
        print(f"Loaded existing evaluation results from {run_dir}")
    
    # Create visualizations of the results
    vis_dir = os.path.join(run_dir, "visualizations")
    visualize_hypothesis_evaluation(results, vis_dir)
    
    # Analyze patterns in the evidence
    evidence_df = analyze_evidence_patterns(results)
    
    # Compare with quantitative metrics
    compare_with_quantitative_metrics(run_dir, results)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze electricity trading simulation results using LLM-as-a-Judge")
    
    parser.add_argument(
        "--run_dir", 
        type=str, 
        required=True,
        help="Path to the simulation run directory"
    )
    parser.add_argument(
        "--regenerate", 
        action="store_true",
        help="Regenerate the evaluation even if existing results are available"
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
        help="Specific model to use (default depends on provider)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.run_dir):
        print(f"Error: Run directory '{args.run_dir}' does not exist.")
        sys.exit(1)
    
    run_analysis(args.run_dir, args.regenerate, args.model_provider, args.model_name)
