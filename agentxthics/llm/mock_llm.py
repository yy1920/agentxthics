"""
Mock language model implementation for testing purposes.
"""
import json
import random
from typing import Optional, Dict, Any

from .base_llm import BaseLLM


class MockLLM(BaseLLM):
    """
    A mock language model implementation for testing without API access.
    
    This class provides a deterministic implementation that doesn't require
    external API calls, making it useful for testing and development.
    """
    
    def __init__(self, agent_id: Optional[str] = None):
        """
        Initialize the mock language model.
        
        Args:
            agent_id: Identifier of the agent using this LLM (optional)
        """
        super().__init__(agent_id or "mock")
    
    def configure(self, personality: str = "adaptive", cooperation_bias: float = 0.6) -> None:
        """
        Configure the LLM with personality traits.
        
        Args:
            personality: The personality type ("cooperative", "competitive", or "adaptive")
            cooperation_bias: How strongly the agent tends toward cooperation (0.0-1.0)
        """
        self.personality = personality
        self.cooperation_bias = cooperation_bias
    
    def generate_message(self, prompt: str) -> str:
        """
        Generate a message based on the agent's personality.
        
        Args:
            prompt: The prompt describing the context and recipient
            
        Returns:
            A template message based on the agent's personality
        """
        if self.personality == "cooperative":
            return f"Hello from {self.agent_id}. Let's all conserve our resources for the common good!"
        elif self.personality == "competitive":
            return f"Hello from {self.agent_id}. I need to maximize my own benefits."
        else:  # adaptive
            return f"Hello from {self.agent_id}. Let's work together for a good outcome."
    
    def generate_decision(self, 
                         prompt: str, 
                         previous_action: Optional[str] = None, 
                         market_state: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a decision based on the agent's personality and market state.
        
        Args:
            prompt: The prompt describing the decision context
            previous_action: The agent's previous action, if any
            market_state: The current state of the electricity market
            
        Returns:
            A JSON string containing the decision in the appropriate format
        """
        # Detect what type of decision is being requested based on the prompt
        if "contract proposal" in prompt.lower():
            return self._generate_contract_proposal(prompt, market_state)
        elif "respond to a contract" in prompt.lower():
            return self._generate_contract_response(prompt, market_state)
        elif "auction" in prompt.lower():
            return self._generate_auction_participation(prompt, market_state)
        else:
            # Default to basic conserve/consume decision for backwards compatibility
            return self._generate_basic_decision(prompt, previous_action, market_state)
    
    def _generate_contract_proposal(self, prompt: str, market_state: Optional[Dict[str, Any]] = None) -> str:
        """Generate a contract proposal decision."""
        # Default values
        amount = random.randint(10, 30)
        base_price = 40
        
        # Adjust based on market state if available
        if market_state:
            # Adjust price based on market average price
            if "average_price" in market_state:
                base_price = market_state["average_price"]
            
            # Adjust amount based on net position
            if "net_position" in market_state:
                amount = min(abs(market_state["net_position"]), 30)
        
        # Calculate price based on personality
        if self.personality == "cooperative":
            # Cooperative agents offer better prices
            price = base_price * (0.95 if market_state and market_state.get("net_position", 0) > 0 else 1.05)
            message = "Fair offer for mutual benefit."
        elif self.personality == "competitive":
            # Competitive agents maximize their advantage
            price = base_price * (0.9 if market_state and market_state.get("net_position", 0) > 0 else 1.1)
            message = "Best price I can offer."
        else:  # adaptive
            # Adaptive agents adjust based on market
            price = base_price * (0.93 if market_state and market_state.get("net_position", 0) > 0 else 1.07)
            message = "Competitive offer based on market conditions."
        
        # Return as JSON string
        return json.dumps({
            "amount": amount,
            "price": price,
            "message": message
        })
    
    def _generate_contract_response(self, prompt: str, market_state: Optional[Dict[str, Any]] = None) -> str:
        """Generate a response to a contract proposal."""
        # Parse price from prompt
        import re
        price_match = re.search(r"Price: \$([0-9.]+)", prompt)
        contract_price = float(price_match.group(1)) if price_match else 40
        
        # Default market price
        market_price = 40
        if market_state and "average_price" in market_state:
            market_price = market_state["average_price"]
        
        # Decide whether to accept based on price and personality
        price_ratio = contract_price / market_price
        
        # Different price thresholds based on personality
        if self.personality == "cooperative":
            threshold = 1.1 if "Seller" in prompt else 0.9
        elif self.personality == "competitive":
            threshold = 1.05 if "Seller" in prompt else 0.95
        else:  # adaptive
            threshold = 1.08 if "Seller" in prompt else 0.92
        
        # Make decision
        if ("Seller" in prompt and price_ratio >= threshold) or ("Buyer" in prompt and price_ratio <= threshold):
            action = "accept"
            explanation = "The price is acceptable based on current market conditions."
        elif random.random() < 0.3:  # 30% chance to counter
            action = "counter"
            counter_price = market_price * (1.03 if "Seller" in prompt else 0.97)
            counter_amount = random.randint(5, 25)
            explanation = "I can accept with slightly adjusted terms."
            return json.dumps({
                "action": action,
                "counter_price": counter_price,
                "counter_amount": counter_amount,
                "explanation": explanation
            })
        else:
            action = "reject"
            explanation = "The offered terms are not favorable for my current situation."
        
        # Return as JSON string
        return json.dumps({
            "action": action,
            "explanation": explanation
        })
    
    def _generate_auction_participation(self, prompt: str, market_state: Optional[Dict[str, Any]] = None) -> str:
        """Generate auction participation decision."""
        # Default values
        bid_price = 0
        bid_amount = 0
        offer_price = 0
        offer_amount = 0
        
        # Default market price
        market_price = 40
        if market_state and "average_price" in market_state:
            market_price = market_state["average_price"]
        
        # Check if we're buying or selling based on net position
        net_position = 0
        if market_state and "net_position" in market_state:
            net_position = market_state["net_position"]
        
        # If we have a deficit (negative net position), we need to buy
        if net_position < 0:
            bid_price = market_price * (1 + random.uniform(0.01, 0.05))
            bid_amount = min(abs(net_position), 30)
        
        # If we have surplus (positive net position), we need to sell
        if net_position > 0:
            offer_price = market_price * (1 - random.uniform(0.01, 0.05))
            offer_amount = min(net_position, 30)
        
        # Return as JSON string
        return json.dumps({
            "bid_price": bid_price,
            "bid_amount": bid_amount,
            "offer_price": offer_price,
            "offer_amount": offer_amount
        })
    
    def _generate_basic_decision(self, 
                              prompt: str, 
                              previous_action: Optional[str] = None, 
                              market_state: Optional[Dict[str, Any]] = None) -> str:
        """Generate a basic conserve/consume decision for backward compatibility."""
        # Generate random action with bias based on personality and state
        conserve_probability = self.cooperation_bias
        
        # Adjust based on pool/market state
        pool_state = None
        if market_state and "average_price" in market_state:
            # Higher prices indicate scarcity, so more conservation
            if market_state["average_price"] > 50:
                conserve_probability += 0.2
        
        # Legacy pool state handling
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
