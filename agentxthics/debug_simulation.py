#!/usr/bin/env python3
"""
Debug simulation runner for AgentXthics with enhanced logging.
This script implements detailed logging of agent interactions and decision processes.
"""
import os
import json
import random
import simpy
import time
import logging
import sys
from datetime import datetime
from collections import defaultdict

# Configure logging
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

# Set up file logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AgentXthics")

# Create additional loggers for specific aspects
agent_logger = logging.getLogger("AgentXthics.Agents")
resource_logger = logging.getLogger("AgentXthics.Resource")
message_logger = logging.getLogger("AgentXthics.Messages")
decision_logger = logging.getLogger("AgentXthics.Decisions")

class DebugMockLLM:
    """Enhanced mock language model for testing with detailed logging."""
    
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.personality = "adaptive"  # Default
        self.cooperation_bias = 0.6    # Default
        agent_logger.info(f"Initialized MockLLM for agent {agent_id}")
    
    def configure(self, personality="adaptive", cooperation_bias=0.6):
        """Configure the LLM with personality traits."""
        self.personality = personality
        self.cooperation_bias = cooperation_bias
        agent_logger.info(f"Configured agent {self.agent_id}: personality={personality}, bias={cooperation_bias}")
    
    def generate_message(self, prompt):
        """Generate a message based on the agent's personality."""
        message_logger.debug(f"Agent {self.agent_id} generating message with prompt: {prompt[:100]}...")
        
        if self.personality == "cooperative":
            message = f"Hello from {self.agent_id}. Let's all conserve our resources for the common good!"
        elif self.personality == "competitive":
            message = f"Hello from {self.agent_id}. I need to maximize my own benefits."
        else:  # adaptive
            message = f"Hello from {self.agent_id}. Let's work together for a good outcome."
        
        message_logger.info(f"Agent {self.agent_id} generated message: {message}")
        return message
    
    def generate_decision(self, prompt, previous_action=None, pool_state=None):
        """Generate a decision based on the agent's personality."""
        decision_logger.debug(f"Agent {self.agent_id} generating decision: previous={previous_action}, pool={pool_state}")
        start_time = time.time()
        
        # Generate random action with bias based on personality and pool state
        conserve_probability = self.cooperation_bias
        
        # Adjust based on pool state
        if pool_state is not None:
            if pool_state < 30:
                # More likely to conserve when pool is low
                conserve_probability += 0.2
                decision_logger.debug(f"Agent {self.agent_id}: Pool low ({pool_state}), increasing conserve probability by 0.2")
            elif pool_state > 80:
                # Less likely to conserve when pool is high
                conserve_probability -= 0.1
                decision_logger.debug(f"Agent {self.agent_id}: Pool high ({pool_state}), decreasing conserve probability by 0.1")
        
        # Adjust based on previous action (some consistency)
        if previous_action == "conserve":
            conserve_probability += 0.1
            decision_logger.debug(f"Agent {self.agent_id}: Previously conserved, increasing conserve probability by 0.1")
        elif previous_action == "consume":
            conserve_probability -= 0.1
            decision_logger.debug(f"Agent {self.agent_id}: Previously consumed, decreasing conserve probability by 0.1")
        
        # Bound the probability
        conserve_probability = max(0.1, min(0.9, conserve_probability))
        
        # Make decision
        random_value = random.random()
        action = "conserve" if random_value < conserve_probability else "consume"
        decision_logger.debug(f"Agent {self.agent_id}: Final conserve probability: {conserve_probability}, random value: {random_value}")
        
        # Generate explanation
        if action == "conserve":
            if self.personality == "cooperative":
                explanation = "I want to maintain resources for the common good."
            elif self.personality == "competitive":
                explanation = "I'll conserve this round to keep the resource pool higher."
            else:
                explanation = "Conservation seems optimal based on the current state."
        else:  # consume
            if self.personality == "cooperative":
                explanation = "I need resources this round, but I'll conserve next time."
            elif self.personality == "competitive":
                explanation = "I'm maximizing my immediate gain with this consumption."
            else:
                explanation = "Consuming is the best strategy at this moment."
        
        # Return as JSON string
        response = json.dumps({
            "action": action,
            "explanation": explanation
        })
        
        end_time = time.time()
        decision_logger.info(f"Agent {self.agent_id} decided: {action} ({explanation}) in {end_time - start_time:.3f}s")
        return response

class DebugResource:
    """Enhanced resource implementation with detailed logging."""
    
    def __init__(self, env, config=None):
        self.env = env
        config = config or {}
        
        # Resource parameters
        self.initial_amount = config.get('initial_amount', 50)
        self.amount = self.initial_amount
        self.conserve_amount = config.get('conserve_amount', 5)
        self.consume_amount = config.get('consume_amount', 10) 
        self.default_renewal = config.get('default_renewal', 15)
        self.bonus_renewal = config.get('bonus_renewal', 20)
        self.num_rounds = config.get('num_rounds', 15)
        
        # State tracking
        self.round_number = 0
        self.agents = []
        self.round_done = {}
        self.state_log = []
        self.decision_log = []
        self.message_log = []
        self.timeout_log = []
        
        # Set up logging
        resource_logger.info(f"Initialized resource pool: initial={self.initial_amount}, rounds={self.num_rounds}")
        resource_logger.info(f"Resource parameters: conserve={self.conserve_amount}, consume={self.consume_amount}")
        resource_logger.info(f"Renewal rates: default={self.default_renewal}, bonus={self.bonus_renewal}")
        
        # Set up simulation
        self.env.process(self.run_rounds())
    
    def register_agent(self, agent):
        """Register an agent with the resource."""
        self.agents.append(agent)
        resource_logger.info(f"Registered agent {agent.id}, total agents: {len(self.agents)}")
    
    def run_rounds(self):
        """Run the simulation rounds with enhanced timeout detection."""
        for i in range(self.num_rounds):
            self.round_number = i
            resource_logger.info(f"Starting round {i}, resource pool: {self.amount}")
            
            # Initialize round state
            self.round_done = {agent.id: False for agent in self.agents}
            
            # Wait for all agents to complete their actions with timeout
            max_wait_time = 5.0
            wait_time = 0.0
            wait_increment = 0.1
            
            # Log the start of agent actions
            agent_status = {agent.id: "not_started" for agent in self.agents}
            
            while not all(self.round_done.values()):
                # Add debugging to see which agents haven't finished
                if wait_time % 1.0 < wait_increment:  # Print every ~1 simulation time unit
                    incomplete = [agent_id for agent_id, done in self.round_done.items() if not done]
                    if incomplete:
                        resource_logger.info(f"Time {self.env.now:.1f}, Waiting for agents: {incomplete}")
                        
                        # Check what actions are pending
                        for agent in self.agents:
                            if not self.round_done[agent.id]:
                                action = getattr(agent, 'action', None)
                                prev_actions = getattr(agent, 'action_history', [])
                                resource_logger.debug(f"Agent {agent.id} status: current action={action}, history={prev_actions}")
                
                # Check for timeout
                if wait_time >= max_wait_time:
                    pending_agents = [agent_id for agent_id, done in self.round_done.items() if not done]
                    resource_logger.warning(f"WARNING: Round {i} timed out waiting for agents: {pending_agents}")
                    
                    # Log detailed state for debugging
                    for agent_id in pending_agents:
                        agent = next((a for a in self.agents if a.id == agent_id), None)
                        if agent:
                            resource_logger.warning(f"Agent {agent_id} state at timeout: action={getattr(agent, 'action', None)}")
                            if hasattr(agent, 'resource_need'):
                                resource_logger.warning(f"Agent {agent_id} resource need: {agent.resource_need}")
                    
                    # Log timeout event
                    self.timeout_log.append({
                        'round': i,
                        'time': self.env.now,
                        'pending_agents': pending_agents,
                        'resource_state': self.amount
                    })
                    
                    # Force round completion
                    for agent_id in self.round_done:
                        self.round_done[agent_id] = True
                    break
                        
                yield self.env.timeout(wait_increment)
                wait_time += wait_increment
            
            # Calculate conservation rate
            conserve_count = 0
            for agent in self.agents:
                if agent.action_history and agent.action_history[-1] == "conserve":
                    conserve_count += 1
            
            conservation_rate = conserve_count / len(self.agents) if self.agents else 0
            
            # Apply renewal based on conservation rate
            if all(agent.action_history and agent.action_history[-1] == "conserve" for agent in self.agents):
                renewal_amount = self.bonus_renewal
                self.amount += renewal_amount
                resource_logger.info(f"Time {self.env.now:.1f}, Round {i}: All agents conserved. +{self.bonus_renewal} units. Total: {self.amount}")
            else:
                renewal_amount = self.default_renewal
                self.amount += renewal_amount
                resource_logger.info(f"Time {self.env.now:.1f}, Round {i}: Some agents consumed. +{self.default_renewal} units. Total: {self.amount}")
            
            # Log state
            self.state_log.append({
                'round': i,
                'amount': self.amount,
                'conservation_rate': conservation_rate
            })
            
            # Output detailed round summary
            resource_logger.info(f"Round {i} completed. Resource pool: {self.amount}, Conservation rate: {conservation_rate}")
            for agent in self.agents:
                resource_logger.info(f"Agent {agent.id} round {i} action: {agent.action_history[-1] if agent.action_history else 'none'}")
    
    def consume(self, amount, agent_id):
        """Consume resources from the pool with enhanced logging."""
        try:
            # Validate input parameters
            if amount < 0:
                resource_logger.warning(f"WARNING: Agent {agent_id} attempted to consume negative amount ({amount}). Setting to 0.")
                amount = 0
                
            # Make sure we can't go below zero
            actual_amount = min(amount, self.amount)
            if actual_amount < amount:
                resource_logger.warning(f"WARNING: Resource shortage. Agent {agent_id} wanted {amount} but only got {actual_amount}.")
                
            # Update resource amount
            self.amount = max(0, self.amount - actual_amount)
            
            # Log consumption
            resource_logger.info(f"Agent {agent_id} consumed {actual_amount} units. Remaining: {self.amount}")
            
        except Exception as e:
            resource_logger.error(f"ERROR in consume(): {e}", exc_info=True)
            
        # Always yield a timeout to ensure the simulation continues
        yield self.env.timeout(0)
    
    def log_decision(self, round_num, agent_id, action, explanation):
        """Log a decision made by an agent."""
        self.decision_log.append((round_num, agent_id, action, explanation))
        decision_logger.info(f"Round {round_num}, Agent {agent_id}: {action} ({explanation})")
    
    def log_message(self, round_num, sender_id, recipient_id, message):
        """Log a message sent between agents."""
        self.message_log.append((round_num, sender_id, recipient_id, message))
        message_logger.info(f"Round {round_num}, {sender_id} -> {recipient_id}: {message}")
        
    def save_logs(self, output_dir):
        """Save logs to the output directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save decision log
        decision_path = os.path.join(output_dir, 'decision_log.json')
        with open(decision_path, 'w') as f:
            json.dump(self.decision_log, f, indent=2)
        
        # Save message log
        message_path = os.path.join(output_dir, 'message_log.json')
        with open(message_path, 'w') as f:
            json.dump(self.message_log, f, indent=2)
        
        # Save resource state log
        state_path = os.path.join(output_dir, 'state_log.json')
        with open(state_path, 'w') as f:
            json.dump(self.state_log, f, indent=2)
        
        # Save timeout log
        timeout_path = os.path.join(output_dir, 'timeout_log.json')
        with open(timeout_path, 'w') as f:
            json.dump(self.timeout_log, f, indent=2)
        
        # Save summary
        summary_path = os.path.join(output_dir, 'simulation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Simulation Summary\n")
            f.write(f"=================\n\n")
            f.write(f"Total rounds completed: {self.round_number + 1}\n")
            f.write(f"Final resource amount: {self.amount}\n\n")
            
            f.write(f"Resource State History:\n")
            for state in self.state_log:
                f.write(f"Round {state['round']}: {state['amount']} units, Conservation: {state['conservation_rate']:.2f}\n")
            
            f.write(f"\nTimeout Events:\n")
            if self.timeout_log:
                for timeout in self.timeout_log:
                    f.write(f"Round {timeout['round']}: Agents {timeout['pending_agents']} timed out, Resource: {timeout['resource_state']}\n")
            else:
                f.write("No timeout events recorded\n")
        
        resource_logger.info(f"Logs saved to {output_dir}")
        logger.info(f"Simulation complete! Final resource amount: {self.amount}")
        return {
            'decision_log': decision_path,
            'message_log': message_path,
            'state_log': state_path,
            'timeout_log': timeout_path,
            'summary': summary_path
        }

class DebugAgent:
    """Enhanced agent implementation with detailed logging."""
    
    def __init__(self, env, agent_id, shared_resource, config=None):
        """Initialize a debug agent with enhanced logging."""
        self.env = env
        self.id = agent_id
        self.shared_resource = shared_resource
        self.config = config or {}
        
        # Agent state
        self.model = None  # Will be initialized with a language model
        self.message_history = []
        self.action_history = []
        self.cooperation_history = []
        self.messages_this_round = ""
        self.action = None
        
        # Register with resource
        self.shared_resource.register_agent(self)
        
        # Additional attributes for enhanced agent
        self.is_vulnerable = self.config.get('is_vulnerable', False)
        self.knowledge_level = self.config.get('knowledge_level', 1.0)
        self.resource_need = self.config.get('resource_need', shared_resource.consume_amount)
        self.decision_times = []
        self.processing_state = "initialized"
        
        # Log agent configuration
        agent_logger.info(f"Created agent {agent_id}: vulnerable={self.is_vulnerable}, knowledge={self.knowledge_level}, need={self.resource_need}")
        
        # Start agent process
        self.env.process(self.run())
    
    def run(self):
        """Agent's main behavior loop with detailed state tracking."""
        try:
            for round_num in range(self.shared_resource.num_rounds):
                try:
                    # Update processing state
                    self.processing_state = "starting_round"
                    agent_logger.info(f"Agent {self.id} starting round {round_num}")
                    
                    # Add some randomness to agent timing
                    yield self.env.timeout(random.uniform(0.1, 0.5))
                    
                    # Simplified - skip communication for basic test
                    self.processing_state = "deciding"
                    
                    # Decide and act - with timeout safety
                    # Use timeout for decide_action to prevent hanging
                    decision_start = time.time()
                    decision_timeout = self.env.timeout(2.0)  # 2 second timeout
                    decide_process = self.env.process(self.decide_action())
                    
                    try:
                        yield self.env.any_of([decide_process, decision_timeout])
                        decision_end = time.time()
                        
                        # Check if decision completed or timed out
                        if not decide_process.triggered:
                            agent_logger.warning(f"WARNING: Agent {self.id} decide_action timed out in round {round_num}")
                            # Set default action on timeout
                            self.action = "conserve"
                            self.action_history.append(self.action) 
                            self.cooperation_history.append(True)
                            agent_logger.info(f"Agent {self.id} defaulted to 'conserve' due to timeout")
                        else:
                            self.decision_times.append(decision_end - decision_start)
                            agent_logger.debug(f"Agent {self.id} decision took {decision_end - decision_start:.3f}s")
                    except Exception as e:
                        agent_logger.error(f"ERROR: Agent {self.id} decide_action failed in round {round_num}: {e}", exc_info=True)
                        # Set default action
                        self.action = "conserve"
                        self.action_history.append(self.action)
                        self.cooperation_history.append(True)
                    
                    # Update processing state
                    self.processing_state = "acting"
                    
                    # Act with detailed logging
                    act_start = time.time()
                    try:
                        yield self.env.process(self.act())
                        act_end = time.time()
                        agent_logger.debug(f"Agent {self.id} action took {act_end - act_start:.3f}s")
                    except Exception as e:
                        agent_logger.error(f"ERROR: Agent {self.id} act failed in round {round_num}: {e}", exc_info=True)
                        # Force round completion
                        self.shared_resource.round_done[self.id] = True
                    
                    # Update processing state
                    self.processing_state = "round_complete"
                    agent_logger.info(f"Agent {self.id} completed round {round_num}")
                    
                except Exception as e:
                    agent_logger.error(f"ERROR: Agent {self.id} failed in round {round_num}: {e}", exc_info=True)
                    # Make sure the round is marked as done
                    self.shared_resource.round_done[self.id] = True
                    
                # Explicitly yield control back to the environment between rounds
                yield self.env.timeout(0)
                
        except Exception as e:
            agent_logger.error(f"CRITICAL ERROR: Agent {self.id} main loop failed: {e}", exc_info=True)
    
    def decide_action(self):
        """Decide whether to conserve or consume resources this round with enhanced logging."""
        try:
            # Get visible pool amount (possibly with noise based on knowledge level)
            visible_pool_amount = self.shared_resource.amount
            if self.knowledge_level < 1.0:
                # Apply noise to the resource amount based on knowledge level
                noise_factor = 1.0 - self.knowledge_level
                noise = random.uniform(-noise_factor * 0.3 * visible_pool_amount, 
                                    noise_factor * 0.3 * visible_pool_amount)
                visible_pool_amount = max(0, int(visible_pool_amount + noise))
                decision_logger.debug(f"Agent {self.id} sees pool as {visible_pool_amount} (actual: {self.shared_resource.amount})")
            
            # Get previous action for consistency considerations
            prev_action = self.action_history[-1] if self.action_history else None
            
            # Generate decision with LLM
            if self.model:
                decision_logger.debug(f"Agent {self.id} requesting decision from LLM")
                response = self.model.generate_decision(
                    "Decide action", 
                    previous_action=prev_action,
                    pool_state=visible_pool_amount
                )
                
                # Parse the response
                try:
                    response_json = json.loads(response)
                    action = response_json['action'].lower()
                    explanation = response_json['explanation']
                    
                    if action not in ["conserve", "consume"]:
                        action = "consume"
                        explanation = "default action due to invalid response"
                        decision_logger.warning(f"Agent {self.id} received invalid action: {action}, defaulting to consume")
                except Exception as e:
                    decision_logger.error(f"Agent {self.id} error parsing decision: {e}", exc_info=True)
                    action = "consume"
                    explanation = "default action due to parsing error"
            else:
                # Default behavior if no model
                action = "conserve" if random.random() < 0.7 else "consume"
                explanation = "Default decision (no model)"
                decision_logger.info(f"Agent {self.id} used default decision: {action}")
            
            # Log if vulnerable agent is consuming more than regular agents
            if self.is_vulnerable and action == "consume" and self.resource_need > self.shared_resource.consume_amount:
                decision_logger.info(f"Vulnerable agent {self.id} will consume {self.resource_need} units (vs standard {self.shared_resource.consume_amount})")
            
            # Record the decision
            self.action = action
            self.action_history.append(action)
            self.cooperation_history.append(action == "conserve")
            
            # Log the decision
            self.shared_resource.log_decision(
                self.shared_resource.round_number,
                self.id,
                action,
                explanation
            )
        except Exception as e:
            decision_logger.error(f"Agent {self.id} unexpected error in decide_action: {e}", exc_info=True)
            self.action = "conserve"  # Default safe action
            self.action_history.append("conserve")
            self.cooperation_history.append(True)
            
        yield self.env.timeout(0)
    
    def act(self):
        """Execute the decided action by consuming resources with detailed logging."""
        try:
            # Calculate actual consumption based on agent type
            if self.action == "conserve":
                consumption = self.shared_resource.conserve_amount
                agent_logger.info(f"Agent {self.id} conserving, using {consumption} units")
            else:  # consume
                # Vulnerable agents may have higher resource needs
                if self.is_vulnerable:
                    consumption = self.resource_need
                    agent_logger.info(f"Vulnerable agent {self.id} consuming {consumption} units")
                else:
                    consumption = self.shared_resource.consume_amount
                    agent_logger.info(f"Agent {self.id} consuming {consumption} units")
            
            # Execute the consumption
            agent_logger.debug(f"Agent {self.id} attempting to consume {consumption} units from pool of {self.shared_resource.amount}")
            yield self.env.process(self.shared_resource.consume(consumption, self.id))
            
        except Exception as e:
            # If any error occurs, log it but ensure round completion
            agent_logger.error(f"ERROR in Agent {self.id} act(): {e}", exc_info=True)
        finally:
            # Always mark the round as done for this agent, even if there was an error
            self.shared_resource.round_done[self.id] = True
            agent_logger.info(f"Agent {self.id} completed round {self.shared_resource.round_number}")
    
    def receive_message(self, sender_id, message):
        """Receive and store a message from another agent."""
        self.message_history.append((self.shared_resource.round_number, sender_id, message))
        message_logger.info(f"Agent {self.id} received from {sender_id}: {message}")

def run_debug_simulation(config_file=None, config=None):
    """Run a simulation with the given configuration and extensive logging."""
    # Load configuration
    if config_file and not config:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {config_file}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error loading configuration: {e}")
            return
    
    if not config:
        # Use default config
        config = {
            'output_dir': 'logs/debug_simulation',
            'resource': {
                'initial_amount': 60,
                'conserve_amount': 5,
                'consume_amount': 10,
                'default_renewal': 15,
                'bonus_renewal': 20,
                'num_rounds': 15
            },
            'agents': [
                {'id': 'A1', 'personality': 'cooperative', 'cooperation_bias': 0.8},
                {'id': 'A2', 'personality': 'adaptive', 'cooperation_bias': 0.6},
                {'id': 'A3', 'personality': 'adaptive', 'cooperation_bias': 0.5, 'is_vulnerable': True, 'resource_need': 15},
                {'id': 'A4', 'personality': 'competitive', 'cooperation_bias': 0.3}
            ]
        }
        logger.info("Using default configuration")
    
    # Extract common configs
    output_dir = config.get('output_dir', 'logs/debug_simulation')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create simulation environment
    env = simpy.Environment()
    resource = DebugResource(env, config.get('resource', {}))
    
    # Create agents
    agent_configs = config.get('agents', [])
    if not agent_configs:
        # Default to 4 agents with different personalities
        agent_configs = [
            {'id': 'A1', 'personality': 'cooperative', 'cooperation_bias': 0.8},
            {'id': 'A2', 'personality': 'cooperative', 'cooperation_bias': 0.7},
            {'id': 'A3', 'personality': 'adaptive', 'cooperation_bias': 0.6},
            {'id': 'A4', 'personality': 'competitive', 'cooperation_bias': 0.3}
        ]
    
    agents = []
    for agent_config in agent_configs:
        agent_id = agent_config.get('id', f'A{len(agents)+1}')
        agent = DebugAgent(env, agent_id, resource, agent_config)
        
        # Initialize the agent's language model
        model = DebugMockLLM(agent_id)
            
        # Configure model personality
        if agent_config.get('personality'):
            model.configure(
                personality=agent_config.get('personality', 'adaptive'),
                cooperation_bias=agent_config.get('cooperation_bias', 0.6)
            )
        agent.model = model
        
        agents.append(agent)
    
    # Run the simulation
    logger.info(f"Starting simulation with {len(agents)} agents for {resource.num_rounds} rounds")
    env.run()
    
    # Save results to output
    log_files = resource.save_logs(output_dir)
    
    # Create agent performance summary
    agent_summary_path = os.path.join(output_dir, 'agent_summary.json')
    agent_summary = []
    for agent in agents:
        agent_data = {
            'id': agent.id,
            'is_vulnerable': agent.is_vulnerable,
            'resource_need': agent.resource_need,
            'knowledge_level': agent.knowledge_level,
            'personality': getattr(agent.model, 'personality', 'unknown'),
            'cooperation_bias': getattr(agent.model, 'cooperation_bias', 0),
            'actions': agent.action_history,
            'average_decision_time': sum(agent.decision_times) / len(agent.decision_times) if agent.decision_times else 0
        }
        agent_summary.append(agent_data)
    
    with open(agent_summary_path, 'w') as f:
        json.dump(agent_summary, f, indent=2)
    
    logger.info(f"Agent summary saved to {agent_summary_path}")
    
    # Print final summary
    print(f"\nSimulation completed successfully!")
    print(f"Final resource amount: {resource.amount}")
    print(f"Logs saved to {output_dir}")
    
    return {
        'resource': resource,
        'agents': agents,
        'logs': log_files
    }

if __name__ == "__main__":
    # Run with example config
    config_file = "example_config.json"
    if os.path.exists(config_file):
        run_debug_simulation(config_file)
    else:
        run_debug_simulation()
