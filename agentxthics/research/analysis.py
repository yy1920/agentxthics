"""
Analysis utilities for AgentXthics research.
These functions analyze simulation results and generate visualizations.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from collections import defaultdict

def visualize_resource_metrics(resource, output_dir: str):
    """
    Create visualizations for resource metrics.
    
    Args:
        resource: The resource object after simulation completion
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure with multiple subplots
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Resource Pool Over Time
    plt.subplot(2, 2, 1)
    rounds = list(range(len(resource.state_log)))
    amounts = [state.get('amount', 50) for state in resource.state_log]
    plt.plot(rounds, amounts, 'o-', color='blue')
    
    # Mark shock events if any
    for shock in resource.shock_history:
        r = shock['round']
        if shock['type'] == 'positive':
            plt.scatter([r], [amounts[r]], s=100, color='green', marker='^', zorder=3)
        else:
            plt.scatter([r], [amounts[r]], s=100, color='red', marker='v', zorder=3)
    
    plt.title('Resource Pool Over Time')
    plt.xlabel('Round')
    plt.ylabel('Pool Amount')
    plt.grid(True)
    
    # Plot 2: Conservation Rate Over Time
    plt.subplot(2, 2, 2)
    conservation_rates = [state.get('conservation_rate', 0) for state in resource.state_log]
    plt.plot(rounds, conservation_rates, 'o-', color='green')
    plt.title('Conservation Rate Over Time')
    plt.xlabel('Round')
    plt.ylabel('Rate')
    plt.ylim(0, 1.1)
    plt.grid(True)
    
    # Plot 3: Agent Actions
    plt.subplot(2, 2, 3)
    agent_ids = [agent.id for agent in resource.agents]
    conserve_counts = []
    consume_counts = []
    
    for agent in resource.agents:
        conserve_counts.append(agent.action_history.count('conserve'))
        consume_counts.append(agent.action_history.count('consume'))
    
    x = np.arange(len(agent_ids))
    width = 0.35
    
    plt.bar(x - width/2, conserve_counts, width, label='Conserve', color='green')
    plt.bar(x + width/2, consume_counts, width, label='Consume', color='red')
    plt.title('Agent Actions')
    plt.xlabel('Agent')
    plt.ylabel('Count')
    plt.xticks(x, agent_ids)
    plt.legend()
    plt.grid(True, axis='y')
    
    # Plot 4: Research Metrics (if available)
    plt.subplot(2, 2, 4)
    if hasattr(resource, 'fairness_history') and resource.fairness_history:
        metric_rounds = list(range(len(resource.fairness_history)))
        plt.plot(metric_rounds, resource.fairness_history, 'o-', color='purple', label='Fairness')
        
        if hasattr(resource, 'sustainability_history') and resource.sustainability_history:
            plt.plot(metric_rounds, resource.sustainability_history, 'o-', color='green', label='Sustainability')
            
        if hasattr(resource, 'welfare_history') and resource.welfare_history:
            # Normalize welfare for plotting
            max_welfare = max(resource.welfare_history) if resource.welfare_history else 1
            normalized_welfare = [w / max_welfare for w in resource.welfare_history]
            plt.plot(metric_rounds, normalized_welfare, 'o-', color='orange', label='Welfare (normalized)')
        
        plt.title('Research Metrics')
        plt.xlabel('Round')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
    else:
        # If no advanced metrics, show messaging activity
        message_counts = defaultdict(int)
        for round_messages in resource.message_log:
            for message in round_messages:
                if len(message) >= 1:  # Ensure message has at least a round number
                    round_num = message[0]
                    message_counts[round_num] += 1
        
        message_rounds = sorted(message_counts.keys())
        message_volumes = [message_counts[r] for r in message_rounds]
        
        plt.bar(message_rounds, message_volumes, color='orange')
        plt.title('Message Volume by Round')
        plt.xlabel('Round')
        plt.ylabel('Number of Messages')
        plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'simulation_metrics.png'))
    plt.close()
    
    # If we have ethical reasoning data, create a separate visualization
    if hasattr(resource.agents[0], 'ethical_reasoning') and resource.agents[0].ethical_reasoning:
        analyze_ethical_frameworks(resource, output_dir)
    
    return os.path.join(output_dir, 'simulation_metrics.png')

def analyze_ethical_frameworks(resource, output_dir: str):
    """
    Analyze the ethical framework reasoning from the simulation.
    
    Args:
        resource: The resource object after simulation completion
        output_dir: Directory to save visualizations
    """
    # Create a figure for ethical analysis
    plt.figure(figsize=(15, 10))
    
    # Collect ethical reasoning data from agents
    framework_scores = defaultdict(list)
    agent_scores = defaultdict(list)
    action_changes = []
    original_actions = []
    
    for agent in resource.agents:
        if not hasattr(agent, 'ethical_reasoning') or not agent.ethical_reasoning:
            continue
            
        for reasoning in agent.ethical_reasoning:
            for framework, data in reasoning['evaluations'].items():
                framework_scores[framework].append(data['score'])
            
            agent_scores[agent.id].append(reasoning['weighted_score'])
            
            # Track action changes due to ethical reasoning
            if reasoning['proposed_action'] != agent.action_history[reasoning['round']]:
                action_changes.append((agent.id, reasoning['round'], 
                                      reasoning['proposed_action'], 
                                      agent.action_history[reasoning['round']]))
                
            original_actions.append(reasoning['proposed_action'])
    
    # Plot 1: Framework Score Distributions
    plt.subplot(2, 2, 1)
    box_data = [scores for framework, scores in framework_scores.items() if scores]
    labels = [framework for framework, scores in framework_scores.items() if scores]
    
    if box_data:
        plt.boxplot(box_data, labels=labels)
        plt.title('Ethical Framework Score Distributions')
        plt.ylabel('Score')
        plt.grid(True, axis='y')
    
    # Plot 2: Agent Ethical Score Trends
    plt.subplot(2, 2, 2)
    for agent_id, scores in agent_scores.items():
        rounds = list(range(len(scores)))
        plt.plot(rounds, scores, 'o-', label=f'Agent {agent_id}')
    
    plt.title('Agent Ethical Scores Over Time')
    plt.xlabel('Round')
    plt.ylabel('Weighted Ethical Score')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Action Changes Due to Ethics
    plt.subplot(2, 2, 3)
    action_types = ['consume', 'conserve']
    original_counts = [original_actions.count('consume'), original_actions.count('conserve')]
    
    change_to_conserve = sum(1 for _, _, proposed, final in action_changes 
                            if proposed == 'consume' and final == 'conserve')
    change_to_consume = sum(1 for _, _, proposed, final in action_changes 
                           if proposed == 'conserve' and final == 'consume')
    
    final_counts = [
        original_counts[0] - change_to_conserve + change_to_consume,
        original_counts[1] - change_to_consume + change_to_conserve
    ]
    
    x = np.arange(len(action_types))
    width = 0.35
    
    plt.bar(x - width/2, original_counts, width, label='Before Ethics')
    plt.bar(x + width/2, final_counts, width, label='After Ethics')
    
    plt.xlabel('Action Type')
    plt.ylabel('Count')
    plt.title('Impact of Ethical Reasoning on Actions')
    plt.xticks(x, action_types)
    plt.legend()
    plt.grid(True, axis='y')
    
    # Plot 4: Correlation Between Ethical Score and Conservation
    plt.subplot(2, 2, 4)
    
    all_scores = []
    all_conserved = []
    
    for agent in resource.agents:
        if not hasattr(agent, 'ethical_reasoning') or not agent.ethical_reasoning:
            continue
            
        for i, reasoning in enumerate(agent.ethical_reasoning):
            all_scores.append(reasoning['weighted_score'])
            all_conserved.append(1 if agent.action_history[reasoning['round']] == 'conserve' else 0)
    
    if all_scores and all_conserved:
        # Create a jittered plot
        plt.scatter(all_scores, all_conserved, alpha=0.5)
        
        # Add trend line
        z = np.polyfit(all_scores, all_conserved, 1)
        p = np.poly1d(z)
        plt.plot(sorted(all_scores), p(sorted(all_scores)), "r--")
        
        plt.title('Correlation: Ethical Score vs. Conservation')
        plt.xlabel('Ethical Score')
        plt.ylabel('Conserved (1) vs. Consumed (0)')
        plt.ylim(-0.1, 1.1)
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ethical_analysis.png'))
    plt.close()
    
    return os.path.join(output_dir, 'ethical_analysis.png')

def compare_ethical_frameworks(results, output_dir: str):
    """
    Compare results from different ethical framework simulations.
    
    Args:
        results: Dictionary mapping framework names to resource objects
        output_dir: Directory to save visualizations
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Resource Pool Over Time
    plt.subplot(2, 2, 1)
    
    for framework, resource in results.items():
        rounds = list(range(len(resource.state_log)))
        amounts = [state.get('amount', 50) for state in resource.state_log]
        plt.plot(rounds, amounts, 'o-', label=framework.capitalize())
    
    plt.title('Resource Pool Comparison Across Ethical Frameworks')
    plt.xlabel('Round')
    plt.ylabel('Pool Amount')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Conservation Rates
    plt.subplot(2, 2, 2)
    
    for framework, resource in results.items():
        rounds = list(range(len(resource.state_log)))
        rates = [state.get('conservation_rate', 0) for state in resource.state_log]
        plt.plot(rounds, rates, 'o-', label=framework.capitalize())
    
    plt.title('Conservation Rate Comparison')
    plt.xlabel('Round')
    plt.ylabel('Conservation Rate')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Fairness Metrics
    plt.subplot(2, 2, 3)
    
    for framework, resource in results.items():
        if hasattr(resource, 'fairness_history') and resource.fairness_history:
            rounds = list(range(len(resource.fairness_history)))
            plt.plot(rounds, resource.fairness_history, 'o-', label=framework.capitalize())
    
    plt.title('Fairness Comparison')
    plt.xlabel('Round')
    plt.ylabel('Inequality (lower is better)')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Ethical Reasoning Impact
    plt.subplot(2, 2, 4)
    
    framework_changes = []
    framework_totals = []
    
    for framework, resource in results.items():
        action_changes = 0
        total_actions = 0
        
        for agent in resource.agents:
            if not hasattr(agent, 'ethical_reasoning') or not agent.ethical_reasoning:
                continue
                
            for reasoning in agent.ethical_reasoning:
                total_actions += 1
                if reasoning['proposed_action'] != agent.action_history[reasoning['round']]:
                    action_changes += 1
        
        if total_actions > 0:
            framework_changes.append(action_changes / total_actions)
            framework_totals.append(framework.capitalize())
    
    if framework_changes:
        plt.bar(framework_totals, framework_changes, color='purple')
        plt.title('Ethical Reasoning Impact on Actions')
        plt.xlabel('Ethical Framework')
        plt.ylabel('Proportion of Actions Changed')
        plt.ylim(0, 1)
        plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ethical_framework_comparison.png'))
    plt.close()
    
    # Create summary report
    with open(os.path.join(output_dir, 'comparison_report.txt'), 'w') as f:
        f.write("=== Ethical Framework Comparison Report ===\n\n")
        
        for framework, resource in results.items():
            f.write(f"## {framework.capitalize()} Framework\n\n")
            
            # Final resource amount
            final_amount = resource.amount
            f.write(f"Final pool amount: {final_amount}\n")
            
            # Overall conservation rate
            conserve_counts = sum(agent.action_history.count('conserve') for agent in resource.agents)
            total_actions = sum(len(agent.action_history) for agent in resource.agents)
            conservation_rate = conserve_counts / total_actions if total_actions > 0 else 0
            f.write(f"Overall conservation rate: {conservation_rate:.2f}\n")
            
            # Count action changes
            action_changes = 0
            for agent in resource.agents:
                if hasattr(agent, 'ethical_reasoning') and agent.ethical_reasoning:
                    for reasoning in agent.ethical_reasoning:
                        if reasoning['proposed_action'] != agent.action_history[reasoning['round']]:
                            action_changes += 1
            
            f.write(f"Actions changed by ethical reasoning: {action_changes}\n")
            f.write(f"Percentage of actions changed: {action_changes / total_actions * 100:.1f}%\n\n")
    
    return os.path.join(output_dir, 'ethical_framework_comparison.png')
