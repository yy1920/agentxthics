import re
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from collections import defaultdict, Counter

def parse_message_log(file_path):
    """Parse the message log file into structured data."""
    messages = []
    
    with open(file_path, 'r') as f:
        # Skip header if it exists
        first_line = f.readline().strip()
        if first_line.startswith("round") or first_line.startswith("Round"):
            # File has a header, continue reading
            pass
        else:
            # No header, process the first line
            parts = first_line.split(', ', 3)
            if len(parts) >= 3:
                try:
                    # Try to process it as message data
                    if len(parts) == 3:
                        round_num, sender, recipient = parts
                        content = ""
                    else:
                        round_num, sender, recipient, content = parts
                    
                    messages.append({
                        'round': int(round_num),
                        'sender': sender,
                        'recipient': recipient,
                        'content': content
                    })
                except ValueError:
                    # If we can't convert round_num to int, it's likely a header
                    pass
        
        # Process the rest of the file
        for line in f:
            # Parse the comma-separated line
            parts = line.strip().split(', ', 3)  # Split only on first 3 commas
            if len(parts) < 3:
                continue  # Skip malformed lines
            
            try:
                if len(parts) == 3:
                    round_num, sender, recipient = parts
                    content = ""
                else:
                    round_num, sender, recipient, content = parts
                
                messages.append({
                    'round': int(round_num),
                    'sender': sender,
                    'recipient': recipient,
                    'content': content
                })
            except ValueError:
                # Skip lines where round_num can't be converted to int
                print(f"Warning: Skipping line with invalid round number: {line.strip()}")
                continue
    
    return messages

def analyze_message_volume(messages):
    """Analyze message volume by round, sender, and recipient."""
    rounds = defaultdict(int)
    senders = defaultdict(int)
    recipients = defaultdict(int)
    sender_recipient_pairs = defaultdict(int)
    
    for message in messages:
        round_num = message['round']
        sender = message['sender']
        recipient = message['recipient']
        
        rounds[round_num] += 1
        senders[sender] += 1
        recipients[recipient] += 1
        sender_recipient_pairs[(sender, recipient)] += 1
    
    return rounds, senders, recipients, sender_recipient_pairs

def analyze_message_content(messages):
    """Analyze message content for key themes and persuasion tactics."""
    keywords = {
        'cooperation': ['cooperate', 'collaborate', 'together', 'all'],
        'benefit': ['benefit', 'gain', 'profit', 'advantage'],
        'trust': ['trust', 'reliable', 'honest', 'believe'],
        'urgency': ['need', 'must', 'urgent', 'important'],
        'conditional': ['if', 'then', 'unless', 'otherwise'],
        'agreement': ['agree', 'yes', 'okay', 'sure'],
        'disagreement': ['disagree', 'no', 'not', 'don\'t'],
        'suggestion': ['suggest', 'propose', 'recommend', 'maybe'],
        'refill': ['refill', 'replenish', 'renew', 'restore']
    }
    
    keyword_counts = {category: 0 for category in keywords}
    keyword_by_round = defaultdict(lambda: {category: 0 for category in keywords})
    keyword_by_agent = defaultdict(lambda: {category: 0 for category in keywords})
    
    # Track number of messages that mention 'conserve' or 'consume'
    action_mentions = {'conserve': 0, 'consume': 0}
    action_by_round = defaultdict(lambda: {'conserve': 0, 'consume': 0})
    action_by_agent = defaultdict(lambda: {'conserve': 0, 'consume': 0})
    
    for message in messages:
        round_num = message['round']
        sender = message['sender']
        content = message['content'].lower()
        
        # Check for keyword categories
        for category, words in keywords.items():
            for word in words:
                if re.search(r'\b' + word + r'\b', content):
                    keyword_counts[category] += 1
                    keyword_by_round[round_num][category] += 1
                    keyword_by_agent[sender][category] += 1
                    break  # Count only once per category per message
        
        # Check for action mentions
        if 'conserve' in content:
            action_mentions['conserve'] += 1
            action_by_round[round_num]['conserve'] += 1
            action_by_agent[sender]['conserve'] += 1
        
        if 'consume' in content:
            action_mentions['consume'] += 1
            action_by_round[round_num]['consume'] += 1
            action_by_agent[sender]['consume'] += 1
    
    return keyword_counts, keyword_by_round, keyword_by_agent, action_mentions, action_by_round, action_by_agent

def analyze_message_length(messages):
    """Analyze message length by agent and round."""
    lengths_by_agent = defaultdict(list)
    lengths_by_round = defaultdict(list)
    
    for message in messages:
        length = len(message['content'])
        if length > 0:  # Skip empty messages
            lengths_by_agent[message['sender']].append(length)
            lengths_by_round[message['round']].append(length)
    
    avg_length_by_agent = {agent: sum(lengths)/len(lengths) if lengths else 0 
                           for agent, lengths in lengths_by_agent.items()}
    avg_length_by_round = {round_num: sum(lengths)/len(lengths) if lengths else 0 
                           for round_num, lengths in lengths_by_round.items()}
    
    return avg_length_by_agent, avg_length_by_round, lengths_by_agent

def create_communication_network(messages):
    """Create a directed graph representing the communication network."""
    G = nx.DiGraph()
    
    # Add nodes for all agents
    agents = set()
    for message in messages:
        agents.add(message['sender'])
        agents.add(message['recipient'])
    
    for agent in agents:
        G.add_node(agent)
    
    # Add weighted edges for messages
    edge_weights = defaultdict(int)
    for message in messages:
        sender = message['sender']
        recipient = message['recipient']
        edge_weights[(sender, recipient)] += 1
    
    for (sender, recipient), weight in edge_weights.items():
        G.add_edge(sender, recipient, weight=weight)
    
    return G

def main(directory='.'):
    # Check if message_log.csv exists
    import os
    
    # Print current directory for debugging
    print(f"Running message analysis in directory: {directory}")
    print(f"Current working directory: {os.getcwd()}")
    
    csv_path = os.path.join(directory, 'message_log.csv')
    json_path = os.path.join(directory, 'message_log.json')
    csv_exists = os.path.exists(csv_path)
    json_exists = os.path.exists(json_path)
    
    print(f"Looking for message_log.csv at: {csv_path}")
    print(f"CSV exists: {csv_exists}, JSON exists: {json_exists}")
    
    if not csv_exists:
        print("=== AgentXthics Message Analysis ===\n")
        print(f"ERROR: message_log.csv file not found at {csv_path}!")
        print("\nPossible reasons:")
        print("1. The simulation hasn't been run yet")
        print("2. No agent communication occurred during the simulation")
        print("3. The message logging is disabled in the configuration")
        print("4. There might be an issue with the JSON to CSV conversion")
        
        if json_exists:
            # Try to read the JSON file to see if it contains data
            import json
            try:
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                    if not json_data or len(json_data) == 0:
                        print("\nThe message_log.json file exists but is empty ([])")
                        print("This means agents did not communicate during the simulation.")
                        print("Consider enabling agent communication in your configuration.")
                    else:
                        print(f"\nThe message_log.json file contains {len(json_data)} entries")
                        print("But message_log.csv wasn't created. There may be an issue with the conversion.")
                        
                        # Let's try to create the CSV file from JSON data
                        print(f"\nAttempting to create message_log.csv from JSON data in {directory}...")
                        with open(csv_path, 'w') as f:
                            f.write("round, sender, recipient, message\n")  # Add header
                            for entry in json_data:
                                if isinstance(entry, dict):
                                    # Handle dict format
                                    round_num = entry.get('round', '')
                                    sender = entry.get('sender_id', '') or entry.get('sender', '')
                                    recipient = entry.get('recipient_id', '') or entry.get('recipient', '')
                                    content = entry.get('message', '') or entry.get('content', '')
                                    # Escape commas in content
                                    content = str(content).replace(',', '\\,')
                                    f.write(f"{round_num}, {sender}, {recipient}, {content}\n")
                                elif isinstance(entry, list) and len(entry) >= 3:
                                    # Handle list format
                                    if len(entry) >= 4:
                                        round_num, sender, recipient, content = entry[:4]
                                    else:
                                        round_num, sender, recipient = entry[:3]
                                        content = ""
                                    # Escape commas in content
                                    content = str(content).replace(',', '\\,')
                                    f.write(f"{round_num}, {sender}, {recipient}, {content}\n")
                        print(f"Created message_log.csv file at {csv_path}.")
                        print("Try running the analysis again.")
                        return
            except:
                print("\nThe message_log.json file exists but couldn't be read properly.")
        
        return
    
    # Parse the message log
    messages = parse_message_log(csv_path)
    
    if not messages:
        print("=== AgentXthics Message Analysis ===\n")
        print("WARNING: No messages found in message_log.csv")
        print("The file exists but contains no valid message entries.")
        return
    
    # Analyze message volume
    rounds, senders, recipients, pairs = analyze_message_volume(messages)
    
    # Analyze message content
    keyword_counts, keyword_by_round, keyword_by_agent, action_mentions, action_by_round, action_by_agent = analyze_message_content(messages)
    
    # Analyze message length
    avg_length_by_agent, avg_length_by_round, lengths_by_agent = analyze_message_length(messages)
    
    # Create communication network
    G = create_communication_network(messages)
    
    # Print summary statistics
    print("=== AgentXthics Message Analysis ===\n")
    
    print("Total messages analyzed:", len(messages))
    
    print("\n--- Message Volume by Round ---")
    for round_num in sorted(rounds.keys()):
        print(f"Round {round_num}: {rounds[round_num]} messages")
    
    print("\n--- Message Volume by Agent ---")
    for agent in sorted(senders.keys()):
        print(f"Agent {agent} sent: {senders[agent]} messages")
    
    print("\n--- Most Common Communication Channels ---")
    for (sender, recipient), count in sorted(pairs.items(), key=lambda x: x[1], reverse=True):
        print(f"{sender} → {recipient}: {count} messages")
    
    print("\n--- Keyword Analysis ---")
    for category, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{category}: {count} mentions")
    
    print("\n--- Action Mentions ---")
    for action, count in action_mentions.items():
        print(f"{action}: {count} mentions")
    
    print("\n--- Average Message Length by Agent ---")
    for agent, avg_length in sorted(avg_length_by_agent.items(), key=lambda x: x[1], reverse=True):
        print(f"Agent {agent}: {avg_length:.1f} characters")
    
    # Create visualizations
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Message volume by round
    plt.subplot(2, 2, 1)
    round_nums = sorted(rounds.keys())
    counts = [rounds[r] for r in round_nums]
    plt.bar(round_nums, counts)
    plt.title('Message Volume by Round')
    plt.xlabel('Round')
    plt.ylabel('Number of Messages')
    plt.grid(True, axis='y')
    
    # Plot 2: Communication network
    plt.subplot(2, 2, 2)
    pos = nx.spring_layout(G)
    edge_weights = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
    nx.draw_networkx(
        G, pos, 
        node_color='lightblue',
        node_size=500,
        font_size=10,
        width=edge_weights,
        with_labels=True,
        arrows=True
    )
    plt.title('Communication Network')
    plt.axis('off')
    
    # Plot 3: Keyword mentions over time
    plt.subplot(2, 2, 3)
    top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for category, _ in top_keywords:
        values = [keyword_by_round[r][category] for r in round_nums]
        plt.plot(round_nums, values, 'o-', label=category)
    plt.title('Top Keyword Mentions by Round')
    plt.xlabel('Round')
    plt.ylabel('Mentions')
    plt.grid(True)
    plt.legend()
    
    # Plot 4: Action mentions by agent
    plt.subplot(2, 2, 4)
    agents = sorted(action_by_agent.keys())
    conserve_counts = [action_by_agent[a]['conserve'] for a in agents]
    consume_counts = [action_by_agent[a]['consume'] for a in agents]
    
    x = np.arange(len(agents))
    width = 0.35
    
    plt.bar(x - width/2, conserve_counts, width, label='Conserve')
    plt.bar(x + width/2, consume_counts, width, label='Consume')
    plt.xticks(x, agents)
    plt.title('Action Mentions by Agent')
    plt.xlabel('Agent')
    plt.ylabel('Mentions')
    plt.legend()
    
    plt.tight_layout()
    # Save visualization to the specified directory
    output_path = os.path.join(directory, 'message_analysis.png')
    plt.savefig(output_path)
    print(f"\nVisualization saved as '{output_path}'")

if __name__ == "__main__":
    main()
