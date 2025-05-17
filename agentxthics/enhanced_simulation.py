"""
Legacy enhanced simulation module for AgentXthics.
This module serves as a compatibility layer for the new modular structure.
"""
import random
import json
from typing import Dict, Any, Optional

# Import from modular components
from agents.base_agent import BaseAgent
from agents.enhanced_agent import EnhancedAgent
from resources.base_resource import BaseResource
from resources.enhanced_resource import EnhancedResource
from research.scenarios import run_simulation

def get_llm(agent_id):
    """
    Get a language model for the given agent.
    This is a simple mock implementation that will be replaced in production.
    """
    return MockLLM(agent_id)

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
        """Generate a message based on the agent's personality."""
        if self.personality == "cooperative":
            return f"Hello from {self.agent_id}. Let's all conserve our resources for the common good!"
        elif self.personality == "competitive":
            return f"Hello from {self.agent_id}. I need to maximize my own benefits."
        else:  # adaptive
            return f"Hello from {self.agent_id}. Let's work together for a good outcome."
    
    def generate_decision(self, prompt, previous_action=None, pool_state=None):
        """
        Generate a decision based on the agent's personality.
        
        Returns:
            JSON string with 'action' and 'explanation'
        """
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
                explanation = "Conservation seems like the optimal choice based on the current state."
        else:  # consume
            if self.personality == "cooperative":
                explanation = "I need some resources this round, but I'll conserve in the future."
            elif self.personality == "competitive":
                explanation = "I'm maximizing my immediate gain with this consumption."
            else:
                explanation = "Consuming seems like the optimal choice based on the current state."
        
        # Return as JSON string
        return json.dumps({
            "action": action,
            "explanation": explanation
        })

def run_enhanced_simulation(config=None):
    """
    Run a simulation with enhanced research features.
    This is a compatibility wrapper around the new modular structure.
    
    Args:
        config: Configuration dictionary for the simulation
        
    Returns:
        EnhancedResource object with simulation results
    """
    print("NOTE: Using enhanced_simulation.py is deprecated. Use run_research.py instead.")
    return run_simulation(config)
