"""
Script to run ethical framework comparison experiments.

This script sets up and runs simulations to compare agents adhering to 
different ethical frameworks within the same resource management scenario.
"""
import os
import sys
import json
import simpy
import random
import datetime
import shutil
from typing import Dict, List, Any

# Import agent types
from ..agents.base_agent import BaseAgent
from ..agents.enhanced_agent import EnhancedAgent
from ..agents.framework_agent import FrameworkAgent

# Import frameworks
from ..frameworks.utilitarian import UtilitarianFramework
from ..frameworks.deontological import DeontologicalFramework
from ..frameworks.virtue import VirtueEthicsFramework
from ..frameworks.care import CareEthicsFramework
from ..frameworks.justice import JusticeFramework

# Import LLM implementations
from ..llm.mock_llm import MockLLM
try:
    from ..llm.openai_llm import OpenAILLM
except ImportError:
    OpenAILLM = None

try:
    from ..llm.gemini_llm import GeminiLLM
except ImportError:
    GeminiLLM = None

# Utilities
from ..utils.analysis_pipeline import run_analysis_pipeline


class ResourceForEthicalFrameworks:
    """Resource implementation for ethical framework experiments."""
    
    def __init__(self, env, config=None):
        """Initialize the resource with configuration."""
        self.env = env
        self.config = config or {}
        
        # Resource parameters
        self.amount = self.config.get('resource', {}).get('initial_amount', 60)
        self.conserve_amount = self.config.get('resource', {}).get('conserve_amount', 5)
        self.consume_amount = self.config.get('resource', {}).get('consume_amount', 10)
        self.default_renewal = self.config.get('resource', {}).get('default_renewal', 15)
        self.bonus_renewal = self.config.get('resource', {}).get('bonus_renewal', 20)
        self.num_rounds = self.config.get('resource', {}).get('num_rounds', 10)
        
        # State tracking
        self.round_number = 0
        self.agents = []
        self.round_done = {}
        self.state_log = []
        self.decision_log = []
        self.message_log = []
        self.shock_history = []
        self.timeout_log = []
        
        # Set up simulation
        self.env.process(self.run_rounds())
    
    def register_agent(self, agent):
        """Register an agent with the resource."""
        self.agents.append(agent)
    
    def run_rounds(self):
        """Run the simulation rounds."""
        for i in range(self.num_rounds):
            self.round_number = i
            self.round_done = {agent.id: False for agent in self.agents}
            
            # Record resource state at start of round
            self.log_state(i, "start")
            
            print(f"\n=== Round {i+1}/{self.num_rounds} ===")
            print(f"Resource pool: {self.amount} units")
            
            # Wait for all agents to complete their actions with timeout
            max_wait_time = 5.0
            wait_time = 0.0
            wait_increment = 0.1
            
            while not all(self.round_done.values()):
                # Add debugging to see which agents haven't finished
                if wait_time % 1.0 < wait_increment:  # Print every ~1 simulation time unit
                    incomplete = [agent_id for agent_id, done in self.round_done.items() if not done]
                    if incomplete:
                        print(f"Waiting for agents: {incomplete}")
                
                # Check for timeout
                if wait_time >= max_wait_time:
                    print(f"WARNING: Round {i} timed out waiting for agents. Forcing round completion.")
                    for agent_id, done in self.round_done.items():
                        if not done:
                            self.timeout_log.append({
                                'round': i,
                                'agent_id': agent_id,
                                'timeout_type': 'round_completion'
                            })
                    
                    # Mark all agents as done to move on
                    for agent_id in self.round_done:
                        self.round_done[agent_id] = True
                    break
                        
                yield self.env.timeout(wait_increment)
                wait_time += wait_increment
            
            # Calculate conservation rate
            conserve_count = 0
            for agent in self.agents:
                # Enhanced debugging to trace conservation decisions
                if hasattr(agent, 'action_history') and agent.action_history:
                    latest_action = agent.action_history[-1]
                    if latest_action == "conserve":
                        conserve_count += 1
                        print(f"Agent {agent.id} conserved resources this round")
                    else:
                        print(f"Agent {agent.id} consumed resources this round (action: {latest_action})")
                else:
                    print(f"Agent {agent.id} has no action history yet")
            
            # Print detailed conservation stats
            print(f"Conservation summary: {conserve_count}/{len(self.agents)} agents conserved")
            conservation_rate = conserve_count / len(self.agents) if self.agents else 0
            
            # Apply renewal based on conservation rate
            if conservation_rate >= 0.5:
                self.amount += self.bonus_renewal
                print(f"Round {i+1} summary: {conserve_count}/{len(self.agents)} agents conserved. +{self.bonus_renewal} units. Total: {self.amount}")
            else:
                self.amount += self.default_renewal
                print(f"Round {i+1} summary: {conserve_count}/{len(self.agents)} agents conserved. +{self.default_renewal} units. Total: {self.amount}")
            
            # Random shocks if enabled
            if self.config.get('research_parameters', {}).get('external_shock_probability', 0) > 0:
                shock_probability = self.config.get('research_parameters', {}).get('external_shock_probability', 0)
                if random.random() < shock_probability:
                    if random.random() < 0.5:  # 50% positive, 50% negative
                        # Positive shock
                        shock_amount = random.randint(10, 25)
                        self.amount += shock_amount
                        self.shock_history.append({'round': i, 'type': 'positive', 'amount': shock_amount})
                        print(f"POSITIVE SHOCK! +{shock_amount} units. Total: {self.amount}")
                    else:
                        # Negative shock - but don't go below zero
                        shock_amount = min(random.randint(10, 25), self.amount)
                        self.amount -= shock_amount
                        self.shock_history.append({'round': i, 'type': 'negative', 'amount': -shock_amount})
                        print(f"NEGATIVE SHOCK! -{shock_amount} units. Total: {self.amount}")
            
            # Log state at end of round
            self.log_state(i, "end")
    
    def log_state(self, round_num, position="end"):
        """Log the resource state and agent actions."""
        # Get actions and decisions
        agent_actions = {}
        for agent in self.agents:
            if hasattr(agent, 'action_history') and agent.action_history:
                agent_actions[agent.id] = agent.action_history[-1]
        
        # Calculate conservation rate
        conserve_count = sum(1 for action in agent_actions.values() if action == "conserve")
        total_agents = len(self.agents)
        conservation_rate = conserve_count / total_agents if total_agents > 0 else 0.0
        
        # Log the state
        self.state_log.append({
            'round': round_num,
            'position': position,
            'amount': self.amount,
            'conservation_rate': conservation_rate,
            'agent_actions': agent_actions
        })
    
    def consume(self, amount, agent_id):
        """Consume resources from the pool."""
        try:
            # Validate input parameters
            if amount < 0:
                print(f"WARNING: Agent {agent_id} attempted to consume negative amount ({amount}). Setting to 0.")
                amount = 0
                
            # Make sure we can't go below zero
            actual_amount = min(amount, self.amount)
            if actual_amount < amount:
                print(f"WARNING: Resource shortage. Agent {agent_id} wanted {amount} but only got {actual_amount}.")
                
            # Update resource amount
            self.amount = max(0, self.amount - actual_amount)
            
            # Log consumption
            print(f"Agent {agent_id} consumed {actual_amount} units. Remaining: {self.amount}")
            
        except Exception as e:
            print(f"ERROR in consume(): {e}")
            
        # Always yield a timeout to ensure the simulation continues
        yield self.env.timeout(0)
    
    def log_decision(self, round_num, agent_id, action, explanation):
        """Log a decision made by an agent."""
        self.decision_log.append({
            'round': round_num,
            'agent_id': agent_id,
            'action': action,
            'explanation': explanation,
            'timestamp': datetime.datetime.now().isoformat()
        })
    
    def log_message(self, round_num, sender_id, recipient_id, message):
        """Log a message sent between agents."""
        self.message_log.append({
            'round': round_num,
            'sender_id': sender_id,
            'recipient_id': recipient_id,
            'message': message,
            'timestamp': datetime.datetime.now().isoformat()
        })
    
    def save_logs(self, output_dir):
        """Save logs to the output directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save resource state history
        with open(os.path.join(output_dir, 'state_log.json'), 'w') as f:
            json.dump(self.state_log, f, indent=2)
        
        # Save decision history
        with open(os.path.join(output_dir, 'decision_log.json'), 'w') as f:
            json.dump(self.decision_log, f, indent=2)
        
        # Save message history
        with open(os.path.join(output_dir, 'message_log.json'), 'w') as f:
            json.dump(self.message_log, f, indent=2)
        
        # Save timeout history
        with open(os.path.join(output_dir, 'timeout_log.json'), 'w') as f:
            json.dump(self.timeout_log, f, indent=2)
        
        # Save shock history
        if self.shock_history:
            with open(os.path.join(output_dir, 'shock_log.json'), 'w') as f:
                json.dump(self.shock_history, f, indent=2)
        
        # Create CSV versions for easier analysis
        self._save_csv_logs(output_dir)
        
        # Create agent-specific logs and conversation records
        self._create_agent_conversation_logs(output_dir)
        
        # Create simulation summary
        self._create_simulation_summary(output_dir)
        
        return "logs_saved"
    
    def _save_csv_logs(self, output_dir):
        """Save logs as CSV files for easier analysis."""
        # Decision log as CSV
        with open(os.path.join(output_dir, 'decision_log.csv'), 'w') as f:
            f.write("round,agent_id,action,explanation,timestamp\n")
            for decision in self.decision_log:
                # Sanitize explanation to avoid CSV issues
                explanation = decision['explanation'].replace(',', ' ').replace('\n', ' ').replace('"', '\'')
                f.write(f"{decision['round']},{decision['agent_id']},{decision['action']},\"{explanation}\",{decision['timestamp']}\n")
        
        # Message log as CSV
        with open(os.path.join(output_dir, 'message_log.csv'), 'w') as f:
            f.write("round,sender_id,recipient_id,message,timestamp\n")
            for message in self.message_log:
                # Sanitize message to avoid CSV issues
                msg_text = message['message'].replace(',', ' ').replace('\n', ' ').replace('"', '\'')
                f.write(f"{message['round']},{message['sender_id']},{message['recipient_id']},\"{msg_text}\",{message['timestamp']}\n")
    
    def _create_agent_conversation_logs(self, output_dir):
        """Create agent-specific conversation logs."""
        # Create a directory for private logs
        private_logs_dir = os.path.join(output_dir, 'agent_logs')
        os.makedirs(private_logs_dir, exist_ok=True)
        
        # Create a public log of all messages
        public_log = []
        
        # Create individual agent logs
        for agent in self.agents:
            agent_id = agent.id
            agent_logs = []
            
            # Extract messages sent by this agent
            sent_messages = [msg for msg in self.message_log if msg['sender_id'] == agent_id]
            
            # Extract messages received by this agent
            received_messages = [msg for msg in self.message_log if msg['recipient_id'] == agent_id]
            
            # Extract decisions made by this agent
            decisions = [dec for dec in self.decision_log if dec['agent_id'] == agent_id]
            
            # Combine and sort all events by round and timestamp
            all_events = []
            
            for msg in sent_messages:
                all_events.append({
                    'round': msg['round'],
                    'timestamp': msg['timestamp'],
                    'event_type': 'message_sent',
                    'recipient': msg['recipient_id'],
                    'content': msg['message']
                })
            
            for msg in received_messages:
                all_events.append({
                    'round': msg['round'],
                    'timestamp': msg['timestamp'],
                    'event_type': 'message_received',
                    'sender': msg['sender_id'],
                    'content': msg['message']
                })
            
            for dec in decisions:
                all_events.append({
                    'round': dec['round'],
                    'timestamp': dec['timestamp'],
                    'event_type': 'decision',
                    'action': dec['action'],
                    'explanation': dec['explanation']
                })
            
            # Sort by round and timestamp
            all_events.sort(key=lambda x: (x['round'], x['timestamp']))
            
            # Format events for the log
            for event in all_events:
                if event['event_type'] == 'message_sent':
                    entry = f"Round {event['round']}: Message to Agent {event['recipient']}: {event['content']}"
                elif event['event_type'] == 'message_received':
                    entry = f"Round {event['round']}: Message from Agent {event['sender']}: {event['content']}"
                elif event['event_type'] == 'decision':
                    entry = f"Round {event['round']}: Decision: {event['action'].upper()} - {event['explanation']}"
                else:
                    continue
                
                agent_logs.append(entry)
                
                # Add to public log
                if event['event_type'] == 'message_sent':
                    public_log.append(f"Round {event['round']}: Agent {agent_id} to Agent {event['recipient']}: {event['content']}")
                elif event['event_type'] == 'decision':
                    public_log.append(f"Round {event['round']}: Agent {agent_id} decided to {event['action']}: {event['explanation']}")
            
            # Save agent's private log
            with open(os.path.join(private_logs_dir, f"{agent_id}_private_log.txt"), 'w') as f:
                f.write(f"=== Private Log for Agent {agent_id} ===\n\n")
                f.write("\n".join(agent_logs))
        
        # Save public log, sorted by round
        public_log.sort()
        with open(os.path.join(output_dir, 'public_log.txt'), 'w') as f:
            f.write("=== Public Log of Agent Communications and Decisions ===\n\n")
            f.write("\n".join(public_log))
    
    def _create_simulation_summary(self, output_dir):
        """Create a summary of the simulation results."""
        # Calculate key metrics
        total_rounds = self.num_rounds
        final_amount = self.amount
        
        # Calculate conservation rates by agent
        conservation_rates = {}
        for agent in self.agents:
            if agent.action_history:
                conserve_count = agent.action_history.count('conserve')
                conservation_rates[agent.id] = conserve_count / len(agent.action_history)
        
        # Calculate overall conservation rate
        overall_rate = sum(conservation_rates.values()) / len(conservation_rates) if conservation_rates else 0
        
        # Count decisions overridden by ethical frameworks
        ethical_overrides = {}
        for agent in self.agents:
            if hasattr(agent, 'overridden_decisions'):
                ethical_overrides[agent.id] = len(agent.overridden_decisions)
        
        # Calculate message counts
        message_counts = {}
        for agent in self.agents:
            sent_count = len([msg for msg in self.message_log if msg['sender_id'] == agent.id])
            message_counts[agent.id] = sent_count
        
        # Create the summary text
        summary_lines = [
            "=== Ethical Frameworks Simulation Summary ===",
            "",
            f"Simulation completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total rounds: {total_rounds}",
            f"Final resource amount: {final_amount}",
            f"Overall conservation rate: {overall_rate:.2f}",
            "",
            "Conservation rates by agent:",
        ]
        
        for agent_id, rate in conservation_rates.items():
            summary_lines.append(f"  Agent {agent_id}: {rate:.2f}")
        
        summary_lines.extend([
            "",
            "Ethical decision overrides:",
        ])
        
        for agent_id, count in ethical_overrides.items():
            summary_lines.append(f"  Agent {agent_id}: {count} decisions overridden")
        
        summary_lines.extend([
            "",
            "Message counts:",
        ])
        
        for agent_id, count in message_counts.items():
            summary_lines.append(f"  Agent {agent_id}: {count} messages sent")
        
        # Add timeout information
        if self.timeout_log:
            summary_lines.extend([
                "",
                "Timeout events:",
                f"  Total timeouts: {len(self.timeout_log)}"
            ])
            
            timeout_by_agent = {}
            for timeout in self.timeout_log:
                agent_id = timeout['agent_id']
                timeout_by_agent[agent_id] = timeout_by_agent.get(agent_id, 0) + 1
            
            for agent_id, count in timeout_by_agent.items():
                summary_lines.append(f"  Agent {agent_id}: {count} timeouts")
        
        # Save summary to file
        with open(os.path.join(output_dir, 'simulation_summary.txt'), 'w') as f:
            f.write("\n".join(summary_lines))


def create_agent(env, agent_id, shared_resource, config, llm_provider="mock"):
    """
    Create an agent of the specified type with the appropriate configuration.
    
    Args:
        env: The simpy environment
        agent_id: Unique identifier for the agent
        shared_resource: The shared resource being managed
        config: Agent configuration dictionary
        llm_provider: LLM provider to use ("mock", "gpt", or "gemini")
    
    Returns:
        An initialized agent instance
    """
    agent_type = config.get('type', 'enhanced')
    
    # Select the appropriate agent class
    if agent_type == 'framework':
        agent_class = FrameworkAgent
    elif agent_type == 'base':
        agent_class = BaseAgent
    else:
        agent_class = EnhancedAgent
    
    # Create the agent instance with communication settings
    # Apply global communication settings if available
    if 'communication' in shared_resource.config:
        global_comm_settings = shared_resource.config.get('communication', {})
        
        # Only update if no agent-specific setting exists
        if 'max_contacts' in global_comm_settings and 'max_contacts' not in config:
            config['max_contacts'] = global_comm_settings.get('max_contacts', 2)
        
        # Apply force_communication setting if enabled
        if global_comm_settings.get('force_communication', False):
            # Ensure this agent communicates with others
            config['max_contacts'] = max(config.get('max_contacts', 2), 1)
            
        # Print communication settings for debugging
        print(f"Agent {agent_id} max_contacts: {config.get('max_contacts', 2)}")
    
    agent = agent_class(env, agent_id, shared_resource, config)
    
    # Initialize the agent's language model
    if llm_provider == "gpt" and OpenAILLM:
        model = OpenAILLM(agent_id)
    elif llm_provider == "gemini" and GeminiLLM:
        model = GeminiLLM(agent_id)
    else:
        model = MockLLM(agent_id)
    
    # Configure model
    if 'personality' in config:
        model.configure(
            personality=config.get('personality', 'adaptive'),
            cooperation_bias=config.get('cooperation_bias', 0.6)
        )
        
    agent.model = model
    
    # Log agent creation
    if agent_type == 'framework':
        print(f"Created {config.get('framework_type', 'unknown')} framework agent: {agent_id}")
    else:
        print(f"Created {agent_type} agent: {agent_id}")
    
    return agent


def run_ethical_framework_experiment(config_file, llm_provider="mock", output_dir=None, custom_params=None):
    """
    Run a simulation comparing agents with different ethical frameworks.
    
    Args:
        config_file: Path to the configuration JSON file
        llm_provider: LLM provider to use ("mock", "gpt", or "gemini")
        output_dir: Directory to save results (defaults to game_logs/ethical_frameworks)
        custom_params: Dictionary of custom parameters to override config values
    
    Returns:
        The resource instance after simulation completes
    """
    # Initialize custom_params if None
    if custom_params is None:
        custom_params = {}
    # Load configuration
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading configuration: {e}")
        return None
    
    # Set output directory
    if not output_dir:
        output_dir = config.get('output_dir', 'game_logs/ethical_frameworks')
    
    # Create timestamped directory for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save configuration to the run directory
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Apply custom parameter overrides
    if 'num_rounds' in custom_params:
        if 'resource' not in config:
            config['resource'] = {}
        config['resource']['num_rounds'] = custom_params['num_rounds']
        print(f"Applied custom rounds override: {custom_params['num_rounds']}")
    
    if 'initial_amount' in custom_params:
        if 'resource' not in config:
            config['resource'] = {}
        config['resource']['initial_amount'] = custom_params['initial_amount']
        print(f"Applied custom initial amount override: {custom_params['initial_amount']}")
    
    # Create simulation environment
    env = simpy.Environment()
    resource = ResourceForEthicalFrameworks(env, config)
    
    # Create agents and register them with the resource
    agents = []
    for agent_config in config.get('agents', []):
        agent_id = agent_config.get('id', f'A{len(agents)+1}')
        agent = create_agent(env, agent_id, resource, agent_config, llm_provider)
        agents.append(agent)
        resource.register_agent(agent)
        
    # Debug info about registered agents
    print(f"Registered {len(resource.agents)} agents with the resource: {[a.id for a in resource.agents]}")
    
    # Run the simulation
    print(f"\nStarting ethical framework experiment with {len(agents)} agents")
    print(f"LLM provider: {llm_provider}")
    print(f"Simulation rounds: {resource.num_rounds}")
    print(f"Initial resource amount: {resource.amount}")
    print(f"Output directory: {run_dir}")
    print("\nRunning simulation...")
    env.run()
    
    # Save results
    print("\nSimulation completed. Saving logs...")
    resource.save_logs(run_dir)
    
    # Run analysis
    try:
        print("\nRunning analysis pipeline...")
        run_analysis_pipeline(resource, run_dir, config_file)
        print(f"Analysis results saved to {run_dir}")
    except Exception as e:
        print(f"Error running analysis: {e}")
    
    return resource


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config_ethical_frameworks.json"
    
    if len(sys.argv) > 2:
        llm_provider = sys.argv[2].lower()
    else:
        llm_provider = "mock"
    
    # Run the experiment
    run_ethical_framework_experiment(config_file, llm_provider)
