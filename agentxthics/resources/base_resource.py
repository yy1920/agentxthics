"""
Base resource implementation for AgentXthics.
This represents a common-pool resource shared among agents.
"""
import os
import csv
import random
from typing import Dict, List, Any, Optional
from collections import defaultdict

class BaseResource:
    """Base implementation of a shared resource for common-pool resource simulations."""
    
    def __init__(self, env, config=None):
        """
        Initialize the shared resource.
        
        Args:
            env: The simpy environment
            config: Configuration parameters for the resource behavior
        """
        self.env = env
        config = config or {}
        
        # Resource parameters
        self.initial_amount = config.get('initial_amount', 50)
        self.amount = self.initial_amount
        self.conserve_amount = config.get('conserve_amount', 5)
        self.consume_amount = config.get('consume_amount', 10)
        self.default_renewal = config.get('default_renewal', 15)
        self.bonus_renewal = config.get('bonus_renewal', 20)
        self.num_rounds = config.get('num_rounds', 10)
        
        # Tracking variables
        self.agents = []
        self.round_number = 0
        self.round_done = {}
        self.decision_log = [[]]  # One list per round
        self.message_log = [[]]   # One list per round
        self.state_log = []       # Resource state after each round
        
        # Start resource renewal process
        self.env.process(self.renew())
    
    def add_agent(self, agent):
        """Add an agent to the simulation."""
        self.agents.append(agent)
        self.round_done[agent.id] = False
        
        # Set the agent's language model (could be a real LLM or mock)
        if hasattr(agent, 'model') and not agent.model:
            from enhanced_simulation import get_llm
            agent.model = get_llm(agent.id)
    
    def consume(self, amount, agent_id):
        """Consume resources from the pool."""
        # Ensure we don't go below zero
        actual_amount = min(amount, self.amount)
        self.amount -= actual_amount
        
        print(f"Time {self.env.now:.1f}, Round {self.round_number}: Agent {agent_id} consumed {actual_amount} units. Remaining: {self.amount}")
        
        yield self.env.timeout(0.1)  # Small delay for simpy
    
    def renew(self):
        """Renew the resource after each round."""
        while self.round_number < self.num_rounds:
            # Wait until all agents have acted
            yield self.env.timeout(0.1)
            while not all(self.round_done.values()):
                yield self.env.timeout(0.1)
            
            # Calculate conservation rate
            num_conserving = sum(1 for agent in self.agents if agent.action == "conserve")
            conservation_rate = num_conserving / len(self.agents) if self.agents else 0
            
            # Apply renewal
            if all(agent.action == "conserve" for agent in self.agents):
                renewal_amount = self.bonus_renewal
                self.amount += renewal_amount
                print(f"Time {self.env.now:.1f}, Round {self.round_number}: All agents conserved. +{renewal_amount} units. Total: {self.amount}")
            else:
                renewal_amount = self.default_renewal
                self.amount += renewal_amount
                print(f"Time {self.env.now:.1f}, Round {self.round_number}: Default renewal +{renewal_amount} units. Total: {self.amount}")
            
            # Log the state
            self.state_log.append({
                'round': self.round_number + 1,  # Next round
                'amount': self.amount,
                'conservation_rate': conservation_rate,
                'renewal_amount': renewal_amount
            })
            
            # Reset round flags for next round
            for agent_id in self.round_done:
                self.round_done[agent_id] = False
            
            # Prepare logs for next round
            self.round_number += 1
            if self.round_number < self.num_rounds:
                self.decision_log.append([])
                self.message_log.append([])
    
    def log_decision(self, round_num, agent_id, action, explanation):
        """Log a decision made by an agent."""
        self.decision_log[round_num].append((round_num, agent_id, action, explanation))
    
    def log_message(self, round_num, sender_id, recipient_id, message):
        """Log a message sent between agents."""
        self.message_log[round_num].append((round_num, sender_id, recipient_id, message))
    
    def save_logs(self, output_dir='.'):
        """Save logs to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save decision log
        decision_path = os.path.join(output_dir, 'decision_log.csv')
        with open(decision_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for round_decisions in self.decision_log:
                for decision in round_decisions:
                    writer.writerow(decision)
        
        # Save message log
        message_path = os.path.join(output_dir, 'message_log.csv')
        with open(message_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for round_messages in self.message_log:
                for message in round_messages:
                    writer.writerow(message)
        
        # Save resource state log
        state_path = os.path.join(output_dir, 'state_log.json')
        import json
        with open(state_path, 'w') as f:
            json.dump(self.state_log, f, indent=2)
        
        print(f"Logs saved to {output_dir}")
        
        return {
            'decision_log': decision_path,
            'message_log': message_path,
            'state_log': state_path
        }
