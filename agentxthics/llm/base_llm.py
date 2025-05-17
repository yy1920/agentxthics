"""
Base language model interface for AgentXthics.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseLLM(ABC):
    """
    Abstract base class for language model implementations.
    
    This class defines the interface that all language model implementations
    must follow to be compatible with AgentXthics agents.
    """
    
    def __init__(self, agent_id: str):
        """
        Initialize the language model.
        
        Args:
            agent_id: Identifier of the agent using this LLM
        """
        self.agent_id = agent_id
        self.personality = "adaptive"  # Default personality
        self.cooperation_bias = 0.6    # Default cooperation bias
    
    @abstractmethod
    def configure(self, personality: str = "adaptive", cooperation_bias: float = 0.6) -> None:
        """
        Configure the LLM with personality traits.
        
        Args:
            personality: The personality type ("cooperative", "competitive", or "adaptive")
            cooperation_bias: How strongly the agent tends toward cooperation (0.0-1.0)
        """
        pass
    
    @abstractmethod
    def generate_message(self, prompt: str) -> str:
        """
        Generate a message to another agent based on the prompt.
        
        Args:
            prompt: The prompt describing the context and recipient
            
        Returns:
            A generated message string
        """
        pass
    
    @abstractmethod
    def generate_decision(self, 
                         prompt: str, 
                         previous_action: Optional[str] = None, 
                         market_state: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a decision based on the prompt and current state.
        
        Args:
            prompt: The prompt describing the decision context
            previous_action: The agent's previous action, if any
            market_state: The current state of the market (or resource pool)
            
        Returns:
            A JSON string containing the decision in the appropriate format
        """
        pass
