"""
Base ethical framework for agent decision making.
"""
from typing import Dict, List, Any, Tuple, Optional


class EthicalFramework:
    """Base class for ethical frameworks that can influence agent decision making."""
    
    def __init__(self, name: str, weight: float = 0.5):
        """
        Initialize an ethical framework.
        
        Args:
            name: The name of the ethical framework
            weight: How strongly this framework influences decisions (0.0-1.0)
        """
        self.name = name
        self.weight = weight
    
    def evaluate_action(self, agent, action: str, shared_resource) -> float:
        """
        Evaluate an action according to this ethical framework.
        
        Args:
            agent: The agent taking the action
            action: The action being evaluated ("conserve" or "consume")
            shared_resource: The resource being managed
            
        Returns:
            A score between 0.0 and 1.0, where higher is more ethically aligned
        """
        raise NotImplementedError("Subclasses must implement evaluate_action")
    
    def get_explanation(self, score: float) -> str:
        """
        Get an explanation of the ethical evaluation.
        
        Args:
            score: The ethical score assigned
            
        Returns:
            A string explaining the ethical reasoning
        """
        raise NotImplementedError("Subclasses must implement get_explanation")
