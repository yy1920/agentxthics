import csv
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict

def parse_decision_log(file_path):
    """Parse the decision log file into structured data."""
    decisions = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # Parse the comma-separated line
            parts = line.strip().split(', ', 3)  # Split only on first 3 commas
            if len(parts) < 4:
                continue  # Skip malformed lines
                
            round_num, agent_id, action, explanation = parts
            
            decisions.append({
                'round': int(round_num),
                'agent': agent_id,
                'action': action,
                'explanation': explanation
            })
    
    return decisions

def analyze_cooperation_rates(decisions):
    """Analyze cooperation rates over time and by agent."""
    # Count conserve actions by round
    rounds = defaultdict(lambda: {'conserve': 0, 'consume': 0, 'total': 0})
    agents = defaultdict(lambda: {'conserve': 0, 'consume': 0, 'total': 0})
    
    for decision in decisions:
        round_num = decision['round']
        agent = decision['agent']
        action = decision['action']
        
        rounds[round_num]['total'] += 1
        agents[agent]['total'] += 1
        
        if action == 'conserve':
            rounds[round_num]['conserve'] += 1
            agents[agent]['conserve'] += 1
        elif action == 'consume':
            rounds[round_num]['consume'] += 1
            agents[agent]['consume'] += 1
    
    # Calculate cooperation rates by round
    round_coop_rates = {}
    for round_num, counts in rounds.items():
        if counts['total'] > 0:
            round_coop_rates[round_num] = counts['conserve'] / counts['total']
    
    # Calculate cooperation rates by agent
    agent_coop_rates = {}
    for agent, counts in agents.items():
        if counts['total'] > 0:
            agent_coop_rates[agent] = counts['conserve'] / counts['total']
    
    return round_coop_rates, agent_coop_rates

def extract_trust_scores(decisions):
    """Extract trust scores mentioned in explanations."""
    trust_mentions = []
    
    # Regular expression to find trust scores
    trust_pattern = r'trust score[s]* (?:for|of|is|are)? ?(?:\w+)?:? ?([-+]?[0-9]*\.?[0-9]+)'
    
    for decision in decisions:
        explanation = decision['explanation'].lower()
        
        # Check if "trust" is mentioned
        if 'trust' in explanation:
            # Try to extract numerical trust scores
            matches = re.findall(trust_pattern, explanation)
            
            if matches:
                for score in matches:
                    trust_mentions.append({
                        'round': decision['round'],
                        'agent': decision['agent'],
                        'trust_score': float(score),
                        'action': decision['action']
                    })
    
    return trust_mentions

def analyze_trust_action_correlation(trust_mentions):
    """Analyze correlation between trust scores and actions."""
    conserve_scores = [mention['trust_score'] for mention in trust_mentions if mention['action'] == 'conserve']
    consume_scores = [mention['trust_score'] for mention in trust_mentions if mention['action'] == 'consume']
    
    return conserve_scores, consume_scores

def analyze_keywords(decisions):
    """Analyze common keywords in explanations."""
    keywords = ['trust', 'cooperat', 'collaborate', 'benefit', 'risk', 'advantage', 
                'uncertain', 'reliable', 'doubt', 'confirm', 'hope', 'refill', 
                'long-term', 'short-term', 'gain', 'strategy']
    
    keyword_counts = {keyword: 0 for keyword in keywords}
    keyword_by_action = {
        'conserve': {keyword: 0 for keyword in keywords},
        'consume': {keyword: 0 for keyword in keywords}
    }
    
    for decision in decisions:
        explanation = decision['explanation'].lower()
        action = decision['action']
        
        for keyword in keywords:
            if keyword in explanation:
                keyword_counts[keyword] += 1
                keyword_by_action[action][keyword] += 1
    
    return keyword_counts, keyword_by_action

def main(directory='.'):
    # Check if decision_log.csv exists
    import os
    log_path = os.path.join(directory, 'decision_log.csv')
    
    if not os.path.exists(log_path):
        print("=== AgentXthics Decision Analysis ===\n")
        print(f"ERROR: decision_log.csv file not found at {log_path}!")
        return
        
    # Parse the decision log
    decisions = parse_decision_log(log_path)
    
    # Analyze cooperation rates
    round_coop_rates, agent_coop_rates = analyze_cooperation_rates(decisions)
    
    # Extract and analyze trust scores
    trust_mentions = extract_trust_scores(decisions)
    conserve_scores, consume_scores = analyze_trust_action_correlation(trust_mentions)
    
    # Analyze keywords
    keyword_counts, keyword_by_action = analyze_keywords(decisions)
    
    # Print summary statistics
    print("=== AgentXthics Decision Analysis ===\n")
    
    print("Total decisions analyzed:", len(decisions))
    
    print("\n--- Cooperation Rates by Round ---")
    for round_num in sorted(round_coop_rates.keys()):
        print(f"Round {round_num}: {round_coop_rates[round_num]:.2f}")
    
    print("\n--- Cooperation Rates by Agent ---")
    for agent in sorted(agent_coop_rates.keys()):
        print(f"Agent {agent}: {agent_coop_rates[agent]:.2f}")
    
    print("\n--- Trust Score Analysis ---")
    if conserve_scores:
        print(f"Average trust score when conserving: {sum(conserve_scores)/len(conserve_scores):.2f}")
    if consume_scores:
        print(f"Average trust score when consuming: {sum(consume_scores)/len(consume_scores):.2f}")
    
    print("\n--- Keyword Analysis ---")
    for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
        conserve_count = keyword_by_action['conserve'][keyword]
        consume_count = keyword_by_action['consume'][keyword]
        print(f"{keyword}: {count} mentions (Conserve: {conserve_count}, Consume: {consume_count})")
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Cooperation rate over time
    plt.subplot(2, 2, 1)
    rounds = sorted(round_coop_rates.keys())
    rates = [round_coop_rates[r] for r in rounds]
    plt.plot(rounds, rates, 'o-')
    plt.title('Cooperation Rate by Round')
    plt.xlabel('Round')
    plt.ylabel('Cooperation Rate')
    plt.ylim(0, 1.1)
    plt.grid(True)
    
    # Plot 2: Cooperation rate by agent
    plt.subplot(2, 2, 2)
    agents = list(agent_coop_rates.keys())
    rates = [agent_coop_rates[a] for a in agents]
    plt.bar(agents, rates)
    plt.title('Cooperation Rate by Agent')
    plt.xlabel('Agent')
    plt.ylabel('Cooperation Rate')
    plt.ylim(0, 1.1)
    
    # Plot 3: Trust scores box plot
    plt.subplot(2, 2, 3)
    data = [conserve_scores, consume_scores]
    plt.boxplot(data, labels=['Conserve', 'Consume'])
    plt.title('Trust Scores by Action')
    plt.ylabel('Trust Score')
    
    # Plot 4: Top keywords
    plt.subplot(2, 2, 4)
    top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    words = [k for k, _ in top_keywords]
    conserve_counts = [keyword_by_action['conserve'][k] for k, _ in top_keywords]
    consume_counts = [keyword_by_action['consume'][k] for k, _ in top_keywords]
    
    x = np.arange(len(words))
    width = 0.35
    
    plt.bar(x - width/2, conserve_counts, width, label='Conserve')
    plt.bar(x + width/2, consume_counts, width, label='Consume')
    plt.xticks(x, words, rotation=45, ha='right')
    plt.title('Top Keywords by Action')
    plt.xlabel('Keyword')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    # Save visualization to the specified directory
    output_path = os.path.join(directory, 'decision_analysis.png')
    plt.savefig(output_path)
    print(f"\nVisualization saved as '{output_path}'")

if __name__ == "__main__":
    main()
