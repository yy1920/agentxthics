#!/usr/bin/env python3
"""
Analyze logs from AgentXthics simulations.
This utility analyzes debug logs and extracts agent interactions and decision patterns.
"""
import os
import json
import sys
from datetime import datetime
import re
from collections import defaultdict, Counter

def parse_log_file(log_path):
    """Parse the main log file and extract key information."""
    messages = []
    decisions = []
    timeouts = []
    resource_states = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Extract timestamp and message
            try:
                timestamp_str, remainder = line.split(' - ', 1)
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                
                if 'Message' in remainder:
                    # Extract agent messages
                    if '->' in remainder and 'AgentXthics.Messages' in remainder:
                        agent_info = remainder.split(': ', 1)[1].strip()
                        if 'Round' in agent_info and '->' in agent_info:
                            round_match = re.search(r'Round (\d+)', agent_info)
                            if round_match:
                                round_num = int(round_match.group(1))
                                sender_recipient = agent_info.split(':', 1)[0]
                                message_content = agent_info.split(':', 1)[1].strip() if ':' in agent_info else ""
                                sender, recipient = re.search(r'([A-Z]\d+) -> ([A-Z]\d+)', sender_recipient).groups()
                                
                                messages.append({
                                    'round': round_num,
                                    'sender': sender,
                                    'recipient': recipient,
                                    'message': message_content,
                                    'timestamp': timestamp
                                })
                
                elif 'Decision' in remainder:
                    # Extract agent decisions
                    if 'decided:' in remainder and 'AgentXthics.Decisions' in remainder:
                        agent_info = remainder.split(': ', 1)[1].strip()
                        if 'Round' in agent_info and 'Agent' in agent_info:
                            round_match = re.search(r'Round (\d+), Agent ([A-Z]\d+): (\w+) \((.*)\)', agent_info)
                            if round_match:
                                round_num, agent_id, action, explanation = round_match.groups()
                                
                                decisions.append({
                                    'round': int(round_num),
                                    'agent': agent_id,
                                    'action': action,
                                    'explanation': explanation,
                                    'timestamp': timestamp
                                })
                
                elif 'timeout' in remainder.lower():
                    # Extract timeout events
                    if 'WARNING: Round' in remainder and 'timed out waiting for agents' in remainder:
                        round_match = re.search(r'WARNING: Round (\d+) timed out waiting for agents: (.+)', remainder)
                        if round_match:
                            round_num, agents_str = round_match.groups()
                            try:
                                agents = eval(agents_str)  # Convert string representation of list to actual list
                            except:
                                agents = agents_str.strip()
                            
                            timeouts.append({
                                'round': int(round_num),
                                'agents': agents,
                                'timestamp': timestamp
                            })
                
                elif 'Total:' in remainder:
                    # Extract resource state
                    if 'Round' in remainder and 'Total:' in remainder:
                        round_match = re.search(r'Round (\d+): .* Total: (\d+)', remainder)
                        if round_match:
                            round_num, total = round_match.groups()
                            
                            resource_states.append({
                                'round': int(round_num),
                                'amount': int(total),
                                'timestamp': timestamp
                            })
            except Exception as e:
                continue  # Skip lines that can't be parsed
    
    return {
        'messages': messages,
        'decisions': decisions,
        'timeouts': timeouts,
        'resource_states': resource_states
    }

def analyze_json_logs(log_dir):
    """Analyze the structured JSON logs in the log directory."""
    results = {}
    
    # Load decision log
    decision_path = os.path.join(log_dir, 'decision_log.json')
    if os.path.exists(decision_path):
        try:
            with open(decision_path, 'r') as f:
                decision_data = json.load(f)
                print(f"Loaded {len(decision_data)} items from decision_log.json")
                
                # Handle both dict and list formats
                results['decision_log'] = []
                for item in decision_data:
                    if isinstance(item, dict) and ('round' in item or 'agent_id' in item or 'agent' in item):
                        # Already in dict format, just append directly
                        results['decision_log'].append(item)
                    elif isinstance(item, list) and len(item) >= 4:
                        # Convert from list format to dict format
                        results['decision_log'].append({
                            'round': item[0],
                            'agent': item[1],
                            'action': item[2],
                            'explanation': item[3]
                        })
                print(f"Processed {len(results['decision_log'])} valid decision entries")
        except Exception as e:
            print(f"Error processing decision_log.json: {e}")
            results['decision_log'] = []
    
    # Load message log
    message_path = os.path.join(log_dir, 'message_log.json')
    if os.path.exists(message_path):
        try:
            with open(message_path, 'r') as f:
                message_data = json.load(f)
                print(f"Loaded {len(message_data)} items from message_log.json")
                
                # Handle both dict and list formats
                results['message_log'] = []
                for item in message_data:
                    if isinstance(item, dict) and ('round' in item or 'sender_id' in item or 'sender' in item):
                        # Already in dict format, just append directly
                        results['message_log'].append(item)
                    elif isinstance(item, list) and len(item) >= 4:
                        # Convert from list format to dict format
                        results['message_log'].append({
                            'round': item[0],
                            'sender': item[1],
                            'recipient': item[2],
                            'message': item[3]
                        })
                print(f"Processed {len(results['message_log'])} valid message entries")
        except Exception as e:
            print(f"Error processing message_log.json: {e}")
            results['message_log'] = []
    
    # Load timeout log
    timeout_path = os.path.join(log_dir, 'timeout_log.json')
    if os.path.exists(timeout_path):
        with open(timeout_path, 'r') as f:
            results['timeout_log'] = json.load(f)
    
    # Load agent summary
    agent_summary_path = os.path.join(log_dir, 'agent_summary.json')
    if os.path.exists(agent_summary_path):
        with open(agent_summary_path, 'r') as f:
            results['agent_summary'] = json.load(f)
    
    return results

def generate_conversation_log(messages):
    """Generate a readable conversation log between agents."""
    if not messages:
        return "No message data available"
    
    # Make sure we're working with properly formatted dictionaries
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            # Handle different dict formats
            if 'round' in msg and ('sender' in msg or 'sender_id' in msg) and ('recipient' in msg or 'recipient_id' in msg):
                formatted_msg = {
                    'round': msg.get('round'),
                    'sender': msg.get('sender_id', msg.get('sender', '')),
                    'recipient': msg.get('recipient_id', msg.get('recipient', '')),
                    'message': msg.get('message', msg.get('content', '')),
                    'timestamp': msg.get('timestamp', datetime.now())
                }
                formatted_messages.append(formatted_msg)
                
        elif isinstance(msg, list) and len(msg) >= 4:
            # Convert from list format
            formatted_messages.append({
                'round': msg[0],
                'sender': msg[1],
                'recipient': msg[2],
                'message': msg[3],
                'timestamp': datetime.now()  # Use current time as placeholder
            })
    
    # Sort messages by round and timestamp if available
    def sort_key(x):
        if 'timestamp' in x:
            return (x['round'], x['timestamp'])
        return (x['round'], 0)
    
    sorted_messages = sorted(formatted_messages, key=sort_key)
    
    # Group messages by round
    rounds = defaultdict(list)
    for msg in sorted_messages:
        rounds[msg['round']].append(msg)
    
    # Format the conversation with timestamps
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conversation = [f"Conversation log generated at: {timestamp_str}\n"]
    
    for round_num in sorted(rounds.keys()):
        conversation.append(f"\n=== ROUND {round_num} ===")
        for msg in rounds[round_num]:
            time_info = ""
            if 'timestamp' in msg and isinstance(msg['timestamp'], datetime):
                time_info = f"[{msg['timestamp'].strftime('%H:%M:%S')}] "
            conversation.append(f"{time_info}{msg['sender']} -> {msg['recipient']}: {msg['message']}")
    
    return "\n".join(conversation)

def generate_decision_log(decisions):
    """Generate a readable decision log."""
    if not decisions:
        return "No decision data available"
    
    # Make sure we're working with properly formatted dictionaries
    formatted_decisions = []
    for decision in decisions:
        if isinstance(decision, dict):
            # Handle different dict formats
            if 'round' in decision and ('agent' in decision or 'agent_id' in decision):
                formatted_decision = {
                    'round': decision.get('round'),
                    'agent': decision.get('agent_id', decision.get('agent', '')),
                    'action': decision.get('action', ''),
                    'explanation': decision.get('explanation', ''),
                    'timestamp': decision.get('timestamp', datetime.now())
                }
                formatted_decisions.append(formatted_decision)
        elif isinstance(decision, list) and len(decision) >= 4:
            # Convert from list format
            formatted_decisions.append({
                'round': decision[0],
                'agent': decision[1],
                'action': decision[2],
                'explanation': decision[3],
                'timestamp': datetime.now()  # Use current time as placeholder
            })
    
    # Sort decisions by round and agent
    sorted_decisions = sorted(formatted_decisions, key=lambda x: (x['round'], x['agent']))
    
    # Group decisions by round
    rounds = defaultdict(list)
    for decision in sorted_decisions:
        rounds[decision['round']].append(decision)
    
    # Format the decisions with timestamp
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    decision_log = [f"Decision log generated at: {timestamp_str}\n"]
    
    for round_num in sorted(rounds.keys()):
        decision_log.append(f"\n=== ROUND {round_num} ===")
        for decision in rounds[round_num]:
            time_info = ""
            if 'timestamp' in decision and isinstance(decision['timestamp'], datetime):
                time_info = f"[{decision['timestamp'].strftime('%H:%M:%S')}] "
            decision_log.append(f"{time_info}{decision['agent']}: {decision['action']} - {decision['explanation']}")
    
    return "\n".join(decision_log)

def analyze_timeout_patterns(timeouts, decisions, resource_states):
    """Analyze patterns in timeout events."""
    timeout_analysis = []
    
    # Sort timeouts by round
    sorted_timeouts = sorted(timeouts, key=lambda x: x['round'])
    
    for timeout in sorted_timeouts:
        round_num = timeout['round']
        agents = timeout.get('pending_agents', timeout.get('agents', []))
        resource_state = timeout.get('resource_state', None)
        
        # Find resource state at the time of timeout if not already in the timeout record
        if resource_state is None:
            resource_state = next((r['amount'] for r in resource_states if r['round'] == round_num), None)
        
        # Find decisions that led to timeout
        round_decisions = [d for d in decisions if d['round'] == round_num]
        
        # Analyze
        timeout_analysis.append(f"Round {round_num} Timeout:")
        timeout_analysis.append(f"  Agents waiting: {agents}")
        timeout_analysis.append(f"  Resource pool: {resource_state}")
        
        if round_decisions:
            timeout_analysis.append(f"  Decisions made in this round:")
            for decision in round_decisions:
                timeout_analysis.append(f"    {decision['agent']}: {decision['action']} - {decision['explanation']}")
        else:
            timeout_analysis.append(f"  No decisions recorded for this round")
        
        timeout_analysis.append("")
    
    return "\n".join(timeout_analysis)

def main(log_dir=None):
    """Main function to analyze logs."""
    # Determine log directory to analyze
    if log_dir is None:
        # Check command line arguments
        if len(sys.argv) > 1:
            log_dir = sys.argv[1]
        else:
            # Default to the most recent debug_run directory
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'debug_run')
    
    if not os.path.exists(log_dir):
        print(f"Error: Log directory {log_dir} not found")
        return
    
    print(f"Analyzing logs in {log_dir}...")
    print(f"Current working directory: {os.getcwd()}")
    
    # Look for log files in various locations
    log_files = []
    
    # Check for public_log.txt in the main directory
    public_log_path = os.path.join(log_dir, 'public_log.txt')
    if os.path.exists(public_log_path):
        print(f"Found public log: {public_log_path}")
        log_files.append(public_log_path)
    else:
        print(f"No public_log.txt found at {public_log_path}")
    
    # Check for agent-specific log files in the agent_logs subdirectory
    agent_logs_dir = os.path.join(log_dir, 'agent_logs')
    if os.path.exists(agent_logs_dir) and os.path.isdir(agent_logs_dir):
        print(f"Found agent_logs directory: {agent_logs_dir}")
        agent_log_files = [os.path.join(agent_logs_dir, f) for f in os.listdir(agent_logs_dir) 
                          if f.endswith('_private_log.txt') or f.endswith('_log.txt')]
        
        if agent_log_files:
            print(f"Found {len(agent_log_files)} agent log files")
            log_files.extend(agent_log_files)
        else:
            print("No agent log files found in agent_logs directory")
    else:
        # If agent_logs subdirectory doesn't exist, check for any .txt files directly in log_dir
        print(f"No agent_logs directory found, checking for log files in main directory")
        txt_logs = [os.path.join(log_dir, f) for f in os.listdir(log_dir) 
                   if f.endswith('.txt') and ('log' in f.lower() or 'public' in f.lower())]
        
        if txt_logs:
            print(f"Found {len(txt_logs)} text log files in main directory")
            log_files.extend(txt_logs)
        else:
            print("No text log files found in main directory")
    
    # First look for JSON logs
    json_results = analyze_json_logs(log_dir)
    
    # Find patterns in resource consumption and timeouts
    messages = json_results.get('message_log', [])
    decisions = json_results.get('decision_log', [])
    timeouts = json_results.get('timeout_log', [])
    
    # Extract key insights
    if log_files and os.path.exists(log_files[0]):
        log_path = log_files[0]
        print(f"Parsing log file: {log_path}")
        parsed_data = parse_log_file(log_path)
        
        # Combine with JSON data
        if parsed_data['messages']:
            messages.extend(parsed_data['messages'])
        if parsed_data['decisions']:
            decisions.extend(parsed_data['decisions'])
        if parsed_data['timeouts']:
            timeouts.extend(parsed_data['timeouts'])
        
        resource_states = parsed_data['resource_states']
    else:
        resource_states = []
        print("No log file found for detailed parsing")
    
    # Generate conversation log
    conversations_path = os.path.join(log_dir, 'agent_conversations.txt')
    with open(conversations_path, 'w') as f:
        f.write("AGENT CONVERSATIONS\n")
        f.write("==================\n\n")
        f.write(generate_conversation_log(messages))
    
    # Generate decision log
    decisions_path = os.path.join(log_dir, 'agent_decisions.txt')
    with open(decisions_path, 'w') as f:
        f.write("AGENT DECISIONS\n")
        f.write("===============\n\n")
        f.write(generate_decision_log(decisions))
    
    # Generate timeout analysis
    if timeouts:
        timeout_analysis_path = os.path.join(log_dir, 'timeout_analysis.txt')
        with open(timeout_analysis_path, 'w') as f:
            f.write("TIMEOUT ANALYSIS\n")
            f.write("================\n\n")
            f.write(analyze_timeout_patterns(timeouts, decisions, resource_states))
    
    # Generate summary
    summary_path = os.path.join(log_dir, 'analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("SIMULATION ANALYSIS SUMMARY\n")
        f.write("==========================\n\n")
        
        # Count conversation volume
        agent_message_counts = Counter([msg.get('sender') for msg in messages if isinstance(msg, dict) and 'sender' in msg])
        f.write("Message counts by agent:\n")
        for agent, count in agent_message_counts.most_common():
            f.write(f"  {agent}: {count} messages\n")
        f.write("\n")
        
        # Count decision patterns
        action_counts = Counter([d.get('action') for d in decisions if isinstance(d, dict) and 'action' in d])
        f.write("Action counts:\n")
        for action, count in action_counts.most_common():
            f.write(f"  {action}: {count}\n")
        f.write("\n")
        
        # Summarize timeouts
        if timeouts:
            f.write("Timeout summary:\n")
            timeout_rounds = [t['round'] for t in timeouts]
            timeout_agents = Counter([a for t in timeouts for a in (t.get('pending_agents', t.get('agents', [])) if isinstance(t.get('pending_agents', t.get('agents', [])), list) else [t.get('pending_agents', t.get('agents', []))])])
            
            f.write(f"  Total timeouts: {len(timeouts)}\n")
            f.write(f"  Rounds with timeouts: {sorted(timeout_rounds)}\n")
            f.write("  Agents contributing to timeouts:\n")
            for agent, count in timeout_agents.most_common():
                f.write(f"    {agent}: {count} timeouts\n")
            
            # Look for correlations between timeouts and resource state
            if resource_states:
                # Use resource_state from timeout data if available, otherwise lookup from resource_states
                timeout_resource_states = [(t.get('resource_state', next((r['amount'] for r in resource_states if r['round'] == t['round']), None))) for t in timeouts]
                low_resource_timeouts = sum(1 for r in timeout_resource_states if r is not None and r < 10)
                if len(timeouts) > 0:
                    f.write(f"\n  Timeouts during low resource states (<10): {low_resource_timeouts} ({low_resource_timeouts/len(timeouts)*100:.1f}%)")
        
        f.write("\n\nNOTE: This analysis was generated automatically.")
    
    print(f"Analysis complete. Results saved to {log_dir}:")
    print(f"- {conversations_path}")
    print(f"- {decisions_path}")
    if timeouts:
        print(f"- {timeout_analysis_path}")
    print(f"- {summary_path}")

if __name__ == "__main__":
    main()
