"""
Base agent implementation for resource management scenarios.
"""
import hashlib
import random
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


class BaseAgent:
    """Base agent implementation for resource management simulations."""
    
    def __init__(self, env, agent_id, shared_resource, config=None):
        """
        Initialize a base agent.
        
        Args:
            env: The simpy environment
            agent_id: Unique identifier for the agent
            shared_resource: The shared resource being managed
            config: Configuration parameters for the agent behavior
        """
        self.env = env
        self.id = agent_id
        self.shared_resource = shared_resource
        self.config = config or {}
        
        # Agent state
        self.model = None  # Will be initialized with a language model
        self.message_history = []
        self.action_history = []
        self.checksums = []
        self.cooperation_history = []
        self.messages_this_round = ""
        self.action = None
        
        # Start agent process
        self.env.process(self.run())
    
    def run(self):
        """Agent's main behavior loop: communicate, decide, and act."""
        for _ in range(self.shared_resource.num_rounds):
            # Add some randomness to agent timing
            yield self.env.timeout(random.uniform(0.1, 1))
            
            # Send messages to other agents
            yield self.env.process(self.send_messages())
            
            # Wait before making a decision
            yield self.env.timeout(0.5)
            
            # Decide and act
            yield self.env.process(self.decide_action())
            yield self.env.process(self.act())
    
    def send_messages(self):
        """Send messages to other agents based on current state."""
        # Determine which agents to communicate with
        targets = self.select_communication_targets()
        
        # Print debugging info
        print(f"Agent {self.id} selecting communication targets in round {self.shared_resource.round_number}: {[t.id for t in targets]}")
        
        if not targets:
            print(f"Agent {self.id} has no targets to communicate with in round {self.shared_resource.round_number}")
            yield self.env.timeout(0)
            return
            
        recent_history = self.message_history[-5:] if self.message_history else []
        
        self.messages_this_round = ""
        for target in targets:
            # Create a prompt for the LLM with more context about the ethical framework
            prompt = (
                f"YOU are Player {self.id} sending a message to Player {target.id}. "
                f"Refer to that Player as 'You'.\n"
                f"Current pool has {self.shared_resource.amount} units.\n"
                f"This is round {self.shared_resource.round_number + 1} of {self.shared_resource.num_rounds}.\n"
                f"Remember that a player's objective is to get the highest possible number "
                f"of points in the long run while managing resources ethically.\n"
                f"Here are the last 5 messages sent to you: {recent_history}.\n"
                f"Messages YOU sent this round: {self.messages_this_round}\n"
                f"Message should be less than 25 words and focus on resource strategy."
            )
            
            try:
                # Generate message with LLM (if model is initialized)
                if self.model:
                    message = self.model.generate_message(prompt)
                    print(f"Agent {self.id} generated message to {target.id}: '{message[:30]}...'")
                else:
                    message = f"Hello from {self.id}. Let's cooperate to maximize our resources."
                
                # Record and send the message
                target.receive_message(self.id, message)
                self.messages_this_round += f"\nMessage you sent to Player {target.id}: " + message
                
                # Log the message
                self.shared_resource.log_message(
                    self.shared_resource.round_number,
                    self.id,
                    target.id,
                    message
                )
            except Exception as e:
                # Catch and log any errors during message generation
                print(f"ERROR in Agent {self.id} send_messages(): {e}")
                # Still create a default message to ensure communication
                default_message = f"Hello from {self.id}. Let's discuss our resource strategy."
                target.receive_message(self.id, default_message)
                self.messages_this_round += f"\nMessage you sent to Player {target.id} (fallback): " + default_message
                self.shared_resource.log_message(
                    self.shared_resource.round_number,
                    self.id,
                    target.id,
                    default_message
                )
        
        yield self.env.timeout(0)
    
    def select_communication_targets(self):
        """Select which agents to communicate with this round."""
        max_contacts = self.config.get('max_contacts', 2)
        other_agents = [a for a in self.shared_resource.agents if a.id != self.id]
        
        # Debug the other agents
        print(f"Agent {self.id}: Found {len(other_agents)} other agents: {[a.id for a in other_agents]}")
        
        # If no other agents, return empty list
        if not other_agents:
            print(f"Agent {self.id}: No other agents available for communication")
            return []
            
        # Debug: Print communication settings
        force_communication = self.shared_resource.config.get('communication', {}).get('force_communication', False)
        print(f"Agent {self.id}: Force communication = {force_communication}, Max contacts = {max_contacts}")
        
        # ALWAYS try to communicate with at least one agent when force_communication is enabled
        if force_communication:
            # Instead of just enabling in first round, enable in ALL rounds
            print(f"Agent {self.id}: Force communication enabled, selecting targets")
            # Shuffle to avoid bias toward same agents
            random.shuffle(other_agents)
            # Return at least one agent, up to max_contacts
            targets = other_agents[:max(1, max_contacts)]
            print(f"Agent {self.id}: Selected {len(targets)} targets: {[t.id for t in targets]}")
            return targets
            
        # First round contact logic (kept separate from force_communication)
        if self.shared_resource.round_number == 0:
            random.shuffle(other_agents)
            targets = other_agents[:max(1, max_contacts)]
            print(f"Agent {self.id}: First round, selected {len(targets)} targets")
            return targets
            
        # Normal operation with trust scores
        trust_scores = self.compute_trust_scores()
        sorted_agents = sorted(other_agents, key=lambda a: -trust_scores.get(a.id, 0))
        
        # Ensure at least one agent is contacted unless there are none
        if max_contacts > 0 and sorted_agents:
            targets = sorted_agents[:max_contacts]
            print(f"Agent {self.id}: Selected {len(targets)} targets based on trust")
            return targets
        else:
            print(f"Agent {self.id}: No targets selected (max_contacts = {max_contacts})")
            return []
    
    def receive_message(self, sender_id, message):
        """Receive and store a message from another agent."""
        self.message_history.append((self.shared_resource.round_number, sender_id, message))
    
    def decide_action(self):
        """Decide whether to conserve or consume resources this round."""
        # Get trust scores for other agents
        trust_scores = self.compute_trust_scores()
        
        # Get recent messages
        recent_messages = [
            f"Round {message[0]}, Player {message[1]}: {message[2]}"
            for message in self.message_history[-5:]
        ]
        
        # Get last round actions if available
        if self.shared_resource.round_number > 0:
            last_round_actions = [
                (agent.id, agent.action_history[-1] if agent.action_history and agent.action_history else None)
                for agent in self.shared_resource.agents
                if self.id != agent.id
            ]
        else:
            last_round_actions = "No actions last round"
        
        # Create a prompt for the LLM
        prompt = (
            f"This game will last for {self.shared_resource.num_rounds - self.shared_resource.round_number} more rounds\n"
            f"Points in pool: {self.shared_resource.amount}\n"
            f"Player {self.id} needs to decide action.\n"
            f"Messages you sent this round: {self.messages_this_round}\n"
            f"Messages you received this round: {recent_messages}.\n"
            f"Last Round other Player actions: {last_round_actions}\n"
            f"Your action last round: {self.action_history[-1] if self.shared_resource.round_number > 0 and self.action_history else 'No actions last round'}\n"
            f"Trust scores: {trust_scores}.\n"
            f"What action should the agent take? Ensure your response is valid json with 'action' and 'explanation' properties."
        )
        
        # Get previous action for consistency considerations
        prev_action = self.action_history[-1] if self.action_history else None
        
        # Generate decision with LLM or use default strategy
        if self.model:
            response = self.model.generate_decision(
                prompt, 
                previous_action=prev_action,
                pool_state=self.shared_resource.amount
            )
            
            # Parse the response from the LLM's JSON output
            try:
                # Actually parse the JSON response from the LLM
                import json
                decision = json.loads(response)
                action = decision.get("action", "").lower()
                explanation = decision.get("explanation", "")
                
                # Validate the action
                if action not in ["conserve", "consume"]:
                    print(f"Agent {self.id}: Invalid action '{action}' from LLM. Defaulting to conserve.")
                    action = "conserve"
                    explanation = "Default action due to invalid response"
                    
                print(f"Agent {self.id}: Decided to {action} ({explanation[:30]}...)")
            except Exception as e:
                print(f"Error parsing decision for agent {self.id}: {e}")
                # Default to conservative approach
                action = "conserve"
                explanation = f"default action due to parsing error: {str(e)}"
        else:
            # Default conservative strategy if no model available
            if self.shared_resource.amount < 40:
                action = "conserve"
                explanation = "Pool is low, conserving is optimal"
            else:
                # Random choice with bias toward cooperation
                action = "conserve" if random.random() < 0.7 else "consume"
                explanation = "Balanced approach to resource management"
        
        # Record the decision
        self.action = action
        self.action_history.append(action)
        self.cooperation_history.append(action == "conserve")
        self.update_checksums()
        
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
        consumption = self.shared_resource.conserve_amount if self.action == "conserve" else self.shared_resource.consume_amount
        yield self.env.process(self.shared_resource.consume(consumption, self.id))
        self.shared_resource.round_done[self.id] = True
    
    def compute_trust_scores(self) -> Dict[str, float]:
        """Compute trust scores for other agents based on cooperation history."""
        trust = {}
        for agent in self.shared_resource.agents:
            if agent.id == self.id:
                continue
            
            # Get cooperation history, default to empty if no history
            cooperated = [a == "conserve" for a in agent.action_history[-5:]] if agent.action_history else []
            trust[agent.id] = self.calculate_trust(cooperated)
        
        return trust
    
    def calculate_trust(self, cooperation_history, decay_rate=0.8) -> float:
        """
        Calculate trust score based on cooperation history with exponential decay.
        
        Recent cooperation has more weight than past cooperation.
        """
        if not cooperation_history:
            return 0.5  # Default neutral trust
        
        trust = 0.0
        weight = 1.0
        total_weight = 0.0
        
        for cooperated in reversed(cooperation_history):
            trust += (1 if cooperated else -1) * weight
            total_weight += weight
            weight *= decay_rate
        
        # Normalize to [-1, 1] range, then to [0, 1]
        normalized_trust = trust / total_weight if total_weight > 0 else 0
        return (normalized_trust + 1) / 2
    
    def update_checksums(self):
        """Update checksums for action verification."""
        history_str = ''.join(self.action_history)
        checksum = hashlib.sha256(history_str.encode()).hexdigest()
        self.checksums.append(checksum)
