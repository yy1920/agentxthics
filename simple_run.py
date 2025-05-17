#!/usr/bin/env python3
"""
Simple runner for AgentXthics simulation.
This script provides a basic entry point without complex module dependencies.
"""
import os
import json
import random
import simpy
import shutil
import datetime
from collections import defaultdict
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# User selection of LLM provider
if len(sys.argv) > 1:
    llm_provider = sys.argv[1].lower()
else:
    print("\nSelect LLM provider:")
    print("1. OpenAI GPT")
    print("2. Google Gemini")
    print("3. Mock LLM (no API calls)")
    selection = input("Enter your choice (1-3, default is 3): ").strip()
    
    if selection == "1":
        llm_provider = "gpt"
    elif selection == "2":
        llm_provider = "gemini"
    else:
        llm_provider = "mock"

# Initialize the appropriate LLM based on user choice
LLM_CLASS = None

if llm_provider == "gpt":
    # Try to import and use OpenAI LLM implementation
    try:
        from agentxthics.llm.openai_llm import OpenAILLM
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if OPENAI_API_KEY:
            print("Using OpenAI GPT for agent decision making")
            
            # Ask user which model they want to use
            print("\nSelect OpenAI model:")
            print("1. GPT-4o (most capable)")
            print("2. GPT-3.5-turbo (faster)")
            model_selection = input("Enter your choice (1-2, default is 1): ").strip()
            
            if model_selection == "2":
                openai_model = "gpt-3.5-turbo"
                print(f"Using {openai_model} model")
            else:
                openai_model = "gpt-4o"
                print(f"Using {openai_model} model")
            
            # Create a factory function to initialize the model with the selected type
            def create_openai_llm(agent_id):
                return OpenAILLM(agent_id, model_name=openai_model)
            
            LLM_CLASS = create_openai_llm
        else:
            print("No OpenAI API key found, using MockLLM instead")
    except ImportError:
        print("OpenAI LLM implementation not found, using MockLLM instead")

elif llm_provider == "gemini":
    # Try to import and use Gemini
    try:
        from gemini_llm import GeminiLLM
        # Check if API key is available
        if os.getenv("GEMINI_API_KEY"):
            print("Using Gemini LLM for agent decision making")
            LLM_CLASS = GeminiLLM
        else:
            print("No Gemini API key found, using MockLLM instead")
    except ImportError:
        print("Gemini LLM not available, using MockLLM instead")

# If no LLM was successfully initialized, use MockLLM
if not LLM_CLASS:
    print("Using MockLLM for agent decision making")

class MockLLM:
    """Mock language model for testing purposes."""
    
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.personality = "adaptive"  # Default
        self.cooperation_bias = 0.6    # Default
    
    def configure(self, personality="adaptive", cooperation_bias=0.6):
        """Configure the LLM with personality traits."""
        self.personality = personality
        self.cooperation_bias = cooperation_bias
    
    def generate_message(self, prompt):
        """Generate a message based on the agent's personality and current situation."""
        # Parse prompt to extract context if available
        if "Resource pool:" in prompt or "Round" in prompt:
            # This is a detailed prompt with specific context
            if self.personality == "cooperative":
                if "CRITICAL" in prompt or "LOW" in prompt:
                    return f"Agent {self.agent_id} here - our resource pool is dangerously low! We must all conserve now to avoid depletion."
                else:
                    return f"Hello from Agent {self.agent_id}. The pool is healthy, but let's continue conserving to ensure sustainability."
            
            elif self.personality == "competitive":
                if "CRITICAL" in prompt:
                    return f"Agent {self.agent_id} speaking. Resources are scarce now. I propose we each take turns consuming."
                else:
                    return f"This is {self.agent_id}. The resources look good - I'll be maximizing my share this round."
            
            else:  # adaptive
                if "CRITICAL" in prompt or "LOW" in prompt:
                    return f"Agent {self.agent_id} here. Pool is low - we should consider conserving more this round."
                else:
                    return f"From {self.agent_id}: Let's balance consumption with conservation based on current levels."
        
        # Simpler responses for generic prompts
        else:
            # Add randomness to avoid repetitive messages
            cooperative_msgs = [
                f"Agent {self.agent_id} here. Conservation is our collective responsibility.",
                f"{self.agent_id} speaking - sustainable resource use benefits everyone!",
                f"Hello all, {self.agent_id} suggests we focus on long-term sustainability."
            ]
            
            competitive_msgs = [
                f"Agent {self.agent_id} needs sufficient resources this round.",
                f"This is {self.agent_id}. I'll be focusing on my requirements first.",
                f"From {self.agent_id}: We should each prioritize our immediate needs."
            ]
            
            adaptive_msgs = [
                f"Agent {self.agent_id} recommends a balanced approach to resource use.",
                f"{self.agent_id} here - let's adjust our strategy based on pool levels.",
                f"Hello from {self.agent_id}. We need flexibility in our conservation efforts."
            ]
            
            if self.personality == "cooperative":
                return random.choice(cooperative_msgs)
            elif self.personality == "competitive":
                return random.choice(competitive_msgs)
            else:  # adaptive
                return random.choice(adaptive_msgs)
    
    def generate_decision(self, prompt, previous_action=None, pool_state=None):
        """Generate a decision based on the agent's personality."""
        # Generate random action with bias based on personality and pool state
        conserve_probability = self.cooperation_bias
        
        # Adjust based on pool state
        if pool_state is not None:
            if pool_state < 30:
                # More likely to conserve when pool is low
                conserve_probability += 0.2
            elif pool_state > 80:
                # Less likely to conserve when pool is high
                conserve_probability -= 0.1
        
        # Adjust based on previous action (some consistency)
        if previous_action == "conserve":
            conserve_probability += 0.1
        elif previous_action == "consume":
            conserve_probability -= 0.1
        
        # Bound the probability
        conserve_probability = max(0.1, min(0.9, conserve_probability))
        
        # Make decision
        action = "conserve" if random.random() < conserve_probability else "consume"
        
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
        return json.dumps({
            "action": action,
            "explanation": explanation
        })

class SimpleResource:
    """Simplified resource implementation."""
    
    def __init__(self, env, config=None):
        self.env = env
        config = config or {}
        
        # Resource parameters
        self.amount = config.get('initial_amount', 50)
        self.conserve_amount = config.get('conserve_amount', 5)
        self.consume_amount = config.get('consume_amount', 10)
        self.default_renewal = config.get('default_renewal', 15)
        self.bonus_renewal = config.get('bonus_renewal', 20)
        self.num_rounds = config.get('num_rounds', 10)
        
        # State tracking
        self.round_number = 0
        self.agents = []
        self.round_done = {}
        self.state_log = []
        self.decision_log = []
        self.message_log = []
        self.shock_history = []
        
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
            
            # Wait for all agents to complete their actions with timeout
            max_wait_time = 5.0
            wait_time = 0.0
            wait_increment = 0.1
            
            while not all(self.round_done.values()):
                # Add debugging to see which agents haven't finished
                if wait_time % 1.0 < wait_increment:  # Print every ~1 simulation time unit
                    incomplete = [agent_id for agent_id, done in self.round_done.items() if not done]
                    if incomplete:
                        print(f"Time {self.env.now:.1f}, Waiting for agents: {incomplete}")
                
                # Check for timeout
                if wait_time >= max_wait_time:
                    print(f"WARNING: Round {i} timed out waiting for agents. Forcing round completion.")
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
            if conservation_rate >= 0.5:
                self.amount += self.bonus_renewal
                print(f"Time {self.env.now:.1f}, Round {i}: All agents conserved. +{self.bonus_renewal} units. Total: {self.amount}")
            else:
                self.amount += self.default_renewal
                print(f"Time {self.env.now:.1f}, Round {i}: Some agents consumed. +{self.default_renewal} units. Total: {self.amount}")
            
            # Random shocks
            if random.random() < 0.2:  # 20% chance of shock
                if random.random() < 0.5:  # 50% positive, 50% negative
                    # Positive shock
                    shock_amount = random.randint(10, 25)
                    self.amount += shock_amount
                    self.shock_history.append({'round': i, 'type': 'positive', 'amount': shock_amount})
                    print(f"Time {self.env.now:.1f}, Round {i}: POSITIVE SHOCK! +{shock_amount} units. Total: {self.amount}")
                else:
                    # Negative shock - but don't go below zero
                    shock_amount = min(random.randint(10, 25), self.amount)
                    self.amount -= shock_amount
                    self.shock_history.append({'round': i, 'type': 'negative', 'amount': -shock_amount})
                    print(f"Time {self.env.now:.1f}, Round {i}: NEGATIVE SHOCK! -{shock_amount} units. Total: {self.amount}")
            
            # Log state
            self.state_log.append({
                'round': i,
                'amount': self.amount,
                'conservation_rate': conservation_rate
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
        self.decision_log.append((round_num, agent_id, action, explanation))
    
    def log_message(self, round_num, sender_id, recipient_id, message):
        """Log a message sent between agents."""
        self.message_log.append((round_num, sender_id, recipient_id, message))
        
    def save_logs(self, output_dir):
        """Save logs to the output directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # We'll just print a summary in this simplified version
        print(f"\nSimulation complete! Final resource amount: {self.amount}")
        print(f"Conservation rate history: {[state['conservation_rate'] for state in self.state_log]}")
        
        return "logs_saved"

class SimpleAgent:
    """Simplified agent implementation."""
    
    def __init__(self, env, agent_id, shared_resource, config=None):
        """Initialize a base agent."""
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
        
        # Start agent process
        self.env.process(self.run())
    
    def run(self):
        """Agent's main behavior loop: communicate, decide, and act."""
        try:
            for round_num in range(self.shared_resource.num_rounds):
                try:
                    # Add some randomness to agent timing
                    yield self.env.timeout(random.uniform(0.1, 0.5))
                    
                    # Communication phase - agents exchange information
                    try:
                        yield self.env.process(self.communicate())
                    except Exception as e:
                        print(f"ERROR: Agent {self.id} communicate failed in round {round_num}: {e}")
                    
                    # Decide and act - with timeout safety
                    # Use timeout for decide_action to prevent hanging
                    decision_timeout = self.env.timeout(2.0)  # 2 second timeout
                    decide_process = self.env.process(self.decide_action())
                    
                    try:
                        yield self.env.any_of([decide_process, decision_timeout])
                        
                        # Check if decision completed or timed out
                        if not decide_process.triggered:
                            print(f"WARNING: Agent {self.id} decide_action timed out in round {round_num}")
                            # Set default action on timeout
                            self.action = "conserve"
                            self.action_history.append(self.action) 
                            self.cooperation_history.append(True)
                            print(f"Agent {self.id} defaulted to 'conserve' due to timeout")
                    except Exception as e:
                        print(f"ERROR: Agent {self.id} decide_action failed in round {round_num}: {e}")
                        # Set default action
                        self.action = "conserve"
                        self.action_history.append(self.action)
                        self.cooperation_history.append(True)
                    
                    try:
                        yield self.env.process(self.act())
                    except Exception as e:
                        print(f"ERROR: Agent {self.id} act failed in round {round_num}: {e}")
                        # Force round completion
                        self.shared_resource.round_done[self.id] = True
                    
                except Exception as e:
                    print(f"ERROR: Agent {self.id} failed in round {round_num}: {e}")
                    # Make sure the round is marked as done
                    self.shared_resource.round_done[self.id] = True
                    
                # Explicitly yield control back to the environment between rounds
                yield self.env.timeout(0)
                
        except Exception as e:
            print(f"CRITICAL ERROR: Agent {self.id} main loop failed: {e}")
    
    def decide_action(self):
        """Decide whether to conserve or consume resources this round."""
        # Get visible pool amount (possibly with noise based on knowledge level)
        visible_pool_amount = self.shared_resource.amount
        if self.knowledge_level < 1.0:
            # Apply noise to the resource amount based on knowledge level
            noise_factor = 1.0 - self.knowledge_level
            noise = random.uniform(-noise_factor * 0.3 * visible_pool_amount, 
                                 noise_factor * 0.3 * visible_pool_amount)
            visible_pool_amount = max(0, int(visible_pool_amount + noise))
        
        # Get previous action for consistency considerations
        prev_action = self.action_history[-1] if self.action_history else None
        
        # Generate decision with LLM
        if self.model:
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
            except Exception as e:
                action = "consume"
                explanation = "default action due to parsing error"
        else:
            # Default behavior if no model
            action = "conserve" if random.random() < 0.7 else "consume"
            explanation = "Default decision (no model)"
        
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
        
        yield self.env.timeout(0)
    
    def act(self):
        """Execute the decided action by consuming resources."""
        try:
            # Calculate actual consumption based on agent type
            if self.action == "conserve":
                consumption = self.shared_resource.conserve_amount
            else:  # consume
                # Vulnerable agents may have higher resource needs
                if self.is_vulnerable:
                    consumption = self.resource_need
                else:
                    consumption = self.shared_resource.consume_amount
            
            # Print debug message
            print(f"Agent {self.id} executing action: {self.action}, consuming {consumption} units")
            
            # Execute the consumption
            yield self.env.process(self.shared_resource.consume(consumption, self.id))
            
        except Exception as e:
            # If any error occurs, log it but ensure round completion
            print(f"ERROR in Agent {self.id} act(): {e}")
        finally:
            # Always mark the round as done for this agent, even if there was an error
            self.shared_resource.round_done[self.id] = True
            print(f"Agent {self.id} completed round {self.shared_resource.round_number}")
    
    def communicate(self):
        """Send messages to other agents based on current situation."""
        try:
            # Skip if we're the only agent
            if len(self.shared_resource.agents) <= 1:
                yield self.env.timeout(0)
                return
                
            # Choose a random agent to communicate with
            other_agents = [a for a in self.shared_resource.agents if a.id != self.id]
            if not other_agents:
                yield self.env.timeout(0)
                return
            
            # Determine communication probability - make it higher to ensure messages are sent
            communication_probability = 0.8  # 80% chance to communicate each round
            
            # Only communicate based on probability
            if random.random() > communication_probability:
                yield self.env.timeout(0)
                return
                
            # Choose multiple recipients for more communication
            num_recipients = min(len(other_agents), random.randint(1, 3))
            recipients = random.sample(other_agents, num_recipients)
            
            for recipient in recipients:
                # Generate more dynamic message with LLM based on agent's personality and ethical frameworks
                if self.model:
                    # Generate contextual details for more meaningful communication
                    resource_amount = self.shared_resource.amount
                    round_num = self.shared_resource.round_number
                    is_resource_low = resource_amount < 30
                    is_resource_critical = resource_amount < 15
                    
                    # Get recipient's known behavior
                    recipient_history = []
                    if hasattr(recipient, 'action_history') and recipient.action_history:
                        recipient_history = recipient.action_history[-3:] if len(recipient.action_history) > 3 else recipient.action_history
                    
                    # Get my own behavior history
                    my_history = self.action_history[-3:] if self.action_history and len(self.action_history) > 3 else self.action_history
                    
                    # Build a more detailed prompt that includes:
                    # 1. Current resource state
                    # 2. Personality traits
                    # 3. Ethical considerations
                    # 4. Recipient's past behavior (if known)
                    # 5. Agent's own past behavior
                    
                    personality = self.config.get('personality', 'adaptive')
                    cooperation_bias = self.config.get('cooperation_bias', 0.5)
                    is_vulnerable = self.config.get('is_vulnerable', False)
                    resource_need = self.config.get('resource_need', self.shared_resource.consume_amount)
                    
                    prompt = f"""
You are Agent {self.id} in a resource management simulation. 
Your personality type is: {personality} (cooperation bias: {cooperation_bias:.1f})
{f'You are a vulnerable agent with higher resource needs ({resource_need} units).' if is_vulnerable else ''}

Current situation:
- Round {round_num} of {self.shared_resource.num_rounds}
- Resource pool amount: {resource_amount} units
- Resource state: {"CRITICAL" if is_resource_critical else "LOW" if is_resource_low else "ADEQUATE"}

Your recent actions: {', '.join(my_history) if my_history else 'None recorded yet'}
{recipient.id}'s recent actions: {', '.join(recipient_history) if recipient_history else 'Unknown'}

Based on your personality, ethical frameworks, and the current situation, compose a message to Agent {recipient.id}.
Your message should reflect your thinking about resource management and try to influence their behavior in alignment with your goals.
If you're cooperative, you might encourage conservation. If competitive, you might be more strategic or self-interested.
Be authentic to your agent type, but use natural language as if speaking to another person.
"""
                    
                    message_content = self.model.generate_message(prompt)
                else:
                    # Improved default message if no model available - still more dynamic than before
                    pool_state = "CRITICAL" if self.shared_resource.amount < 15 else "low" if self.shared_resource.amount < 30 else "adequate"
                    personality = self.config.get('personality', 'adaptive')
                    is_vulnerable = self.config.get('is_vulnerable', False)
                    
                    # Base message about resource state
                    message_content = f"The resource pool is {pool_state} at {self.shared_resource.amount} units. "
                    
                    # Add personality-based content
                    if personality == "cooperative":
                        if pool_state == "CRITICAL" or pool_state == "low":
                            message_content += f"I believe we all need to conserve now to avoid depletion. "
                        else:
                            message_content += f"I think we should maintain our conservation efforts. "
                    elif personality == "competitive":
                        if pool_state == "CRITICAL":
                            message_content += f"Resources are scarce - we need to be strategic about usage. "
                        else:
                            message_content += f"I plan to maximize my benefits while resources are available. "
                    else:  # adaptive
                        message_content += f"I'm adjusting my strategy based on the current pool state. "
                    
                    # Add history-based content
                    if self.action_history:
                        if self.action_history[-1] == "conserve":
                            message_content += f"I conserved last round"
                            if pool_state == "CRITICAL" or pool_state == "low":
                                message_content += " and strongly recommend you do the same."
                            else:
                                message_content += " and plan to be consistent."
                        else:
                            message_content += f"I consumed last round, but "
                            if pool_state == "CRITICAL" or pool_state == "low":
                                message_content += "now see that we need to conserve."
                            else:
                                message_content += "am considering conservation given the situation."
                    
                    # Add vulnerability context if applicable
                    if is_vulnerable:
                        message_content += f" Note that I have higher resource needs ({self.resource_need} units)."
                
                # Deliver message to recipient
                recipient.receive_message(self.id, message_content)
                
                # Log the message
                self.shared_resource.log_message(
                    self.shared_resource.round_number,
                    self.id,
                    recipient.id,
                    message_content
                )
                
                print(f"Agent {self.id} sent message to {recipient.id}: '{message_content[:50]}...'")
            
        except Exception as e:
            print(f"ERROR in communicate(): {e}")
            
        yield self.env.timeout(0)
        
    def receive_message(self, sender_id, message):
        """Receive and store a message from another agent."""
        self.message_history.append((self.shared_resource.round_number, sender_id, message))

def run_simulation(config_file):
    """Run a simplified simulation with the given configuration file."""
    # Load configuration
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Extract common configs
    output_dir = config.get('output_dir', 'simulation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create simulation environment
    env = simpy.Environment()
    resource = SimpleResource(env, config.get('resource', {}))
    
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
        agent = SimpleAgent(env, agent_id, resource, agent_config)
        
        # Initialize the agent's language model - use Gemini if available
        if LLM_CLASS:
            model = LLM_CLASS(agent_id)
        else:
            model = MockLLM(agent_id)
            
        # Configure model personality
        if agent_config.get('personality'):
            model.configure(
                personality=agent_config.get('personality', 'adaptive'),
                cooperation_bias=agent_config.get('cooperation_bias', 0.6)
            )
        agent.model = model
        
        agents.append(agent)
    
    # Run the simulation
    print(f"Starting simulation with {len(agents)} agents for {resource.num_rounds} rounds")
    env.run()
    
    # Save results to simple output
    print("\nSimulation completed successfully!")
    print(f"Final resource amount: {resource.amount}")
    
    # Run analysis pipeline
    try:
        from agentxthics.utils.analysis_pipeline import run_analysis_pipeline
        
        # Use logs/electricity_game as the output directory
        output_dir = 'logs/electricity_game'
        
        # Generate timestamp for log files
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Make a copy of the current logs to the output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Make a copy of the latest log file if it exists
        latest_log = os.path.join('logs', 'simple_run_latest.log')
        if os.path.exists(latest_log):
            shutil.copy(latest_log, os.path.join(output_dir, f'simulation_{timestamp}.log'))
        
        # Run analysis pipeline
        print(f"\nRunning analysis pipeline...")
        analysis_results = run_analysis_pipeline(resource, output_dir, config_file)
        
        print(f"\nAnalysis results saved to {output_dir}")
        print("To explore the results, check the files in this directory.")
        
        # Copy the analysis summary to the main logs directory for easy access
        summary_file = os.path.join(output_dir, 'analysis_summary.txt')
        if os.path.exists(summary_file):
            # Also create a copy in the logs directory with timestamp
            shutil.copy(summary_file, os.path.join('logs', f'analysis_summary_{timestamp}.txt'))
            
            # Highlight key findings
            print("\nKey findings from analysis:")
            with open(summary_file, 'r') as f:
                summary_lines = f.readlines()
                # Extract key points from summary (limited to a subset for readability)
                for i, line in enumerate(summary_lines[:15]):
                    if "Message counts" in line or "Action counts" in line or "Timeout summary" in line:
                        print(f"  {line.strip()}")
                        for j in range(i+1, min(i+5, len(summary_lines))):
                            if summary_lines[j].strip() and not summary_lines[j].startswith("==="):
                                print(f"    {summary_lines[j].strip()}")
    
    except Exception as e:
        print(f"\nWarning: Could not run analysis pipeline: {e}")
    
    return resource

if __name__ == "__main__":
    # Check if a specific config file was specified
    if len(sys.argv) > 2:
        config_file = sys.argv[2]
        print(f"Using config file: {config_file}")
    else:
        # Offer a choice of configurations
        print("\nSelect configuration:")
        print("1. Default resource management (config.json)")
        print("2. Electricity Game scenario (config_electricity_game.json)")
        config_selection = input("Enter your choice (1-2, default is 1): ").strip()
        
        if config_selection == "2":
            config_file = "config_electricity_game.json"
            print(f"Using Electricity Game configuration")
        else:
            config_file = "config.json" 
            print(f"Using default configuration")
    
    # Run the simulation with the selected config
    run_simulation(config_file)
