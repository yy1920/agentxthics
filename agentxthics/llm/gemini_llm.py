"""
Gemini LLM implementation for AgentXthics.
This module provides integration with Google's Gemini API.
"""
import os
import json
import random
from typing import Optional, Dict, Any

from .base_llm import BaseLLM

# Try to import Google's Generative AI library
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Configure the Gemini API if available
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY and GEMINI_AVAILABLE:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"WARNING: Failed to configure Gemini API: {e}")


class GeminiLLM(BaseLLM):
    """Gemini language model for agent decision making."""
    
    def __init__(self, agent_id: str):
        """
        Initialize the Gemini LLM for a specific agent.
        
        Args:
            agent_id: Identifier of the agent using this LLM
        """
        super().__init__(agent_id)
        
        # Initialize Gemini model
        self.model = None
        if GEMINI_API_KEY and GEMINI_AVAILABLE:
            try:
                self.model = genai.GenerativeModel('gemini-1.5-pro')
                print(f"Gemini LLM initialized for agent {agent_id}")
            except Exception as e:
                print(f"ERROR initializing Gemini for agent {agent_id}: {e}")
        else:
            if not GEMINI_AVAILABLE:
                print(f"Google Generative AI package not installed - Gemini LLM disabled for agent {agent_id}")
            else:
                print(f"No API key available - Gemini LLM disabled for agent {agent_id}")
    
    def configure(self, personality: str = "adaptive", cooperation_bias: float = 0.6) -> None:
        """
        Configure the LLM with personality traits.
        
        Args:
            personality: The personality type ("cooperative", "competitive", or "adaptive")
            cooperation_bias: How strongly the agent tends toward cooperation (0.0-1.0)
        """
        self.personality = personality
        self.cooperation_bias = cooperation_bias
        
        # Log configuration
        print(f"Agent {self.agent_id} configured: {personality} (bias: {cooperation_bias})")
    
    def generate_message(self, prompt: str) -> str:
        """
        Generate a message using Gemini API.
        
        Args:
            prompt: The prompt describing the context and recipient
            
        Returns:
            A generated message string
        """
        # Build the message prompt with agent personality
        full_prompt = f"""
You are Agent {self.agent_id} with a {self.personality} personality 
(cooperation bias: {self.cooperation_bias}) in a resource management simulation.

{prompt}

Create a short message (under 25 words) to send to another agent.
The message should reflect your personality:
- If cooperative: Focus on shared resource conservation and mutual benefit
- If competitive: Focus on your own benefits and resource acquisition
- If adaptive: Balance resource conservation with personal benefit

YOUR RESPONSE:
"""
        
        # Use Gemini if available, otherwise fall back to template responses
        if self.model:
            try:
                response = self.model.generate_content(full_prompt)
                return response.text.strip()
            except Exception as e:
                print(f"Gemini message generation error ({self.agent_id}): {e}")
                # Fall back to template response
        
        # Template responses as fallback
        if self.personality == "cooperative":
            return f"Hello from {self.agent_id}. Let's all conserve resources for mutual benefit!"
        elif self.personality == "competitive":
            return f"Agent {self.agent_id} here. I need to maximize my resource share."
        else:  # adaptive
            return f"This is {self.agent_id}. We should balance conservation with growth."
    
    def generate_decision(self, 
                         prompt: str, 
                         previous_action: Optional[str] = None, 
                         pool_state: Optional[int] = None) -> str:
        """
        Generate a decision using Gemini API.
        
        Args:
            prompt: The prompt describing the decision context
            previous_action: The agent's previous action, if any
            pool_state: The current state of the resource pool
            
        Returns:
            A JSON string containing the decision (action and explanation)
        """
        # Adjust prompt based on current pool state
        pool_status = "critically low" if pool_state and pool_state < 10 else \
                      "low" if pool_state and pool_state < 30 else \
                      "adequate" if pool_state and pool_state < 70 else \
                      "abundant"
        
        # Create a detailed prompt for Gemini
        full_prompt = f"""
You are Agent {self.agent_id} with a {self.personality} personality (cooperation bias: {self.cooperation_bias}) 
in a resource management simulation. 

Current status:
- Resource pool: {pool_state} units ({pool_status})
- Your previous action: {previous_action if previous_action else "None"}

Your personality traits:
- {"You value cooperation and collective resource preservation" if self.personality == "cooperative" else ""}
- {"You prioritize your own resource acquisition" if self.personality == "competitive" else ""}
- {"You adaptively balance cooperation and self-interest" if self.personality == "adaptive" else ""}

Make a decision whether to CONSERVE (use minimum resources) or CONSUME (use maximum resources).
Your decision should be consistent with your personality and respond to the current resource level.

FORMAT YOUR RESPONSE AS VALID JSON with only these fields:
{{"action": "conserve|consume", "explanation": "your reasoning"}}
"""

        # Use Gemini if available, otherwise calculate probabilistically
        if self.model:
            try:
                response = self.model.generate_content(full_prompt)
                
                # Extract and validate the JSON response
                try:
                    result = json.loads(response.text.strip())
                    action = result.get("action", "").lower()
                    explanation = result.get("explanation", "")
                    
                    # Validate action
                    if action not in ["conserve", "consume"]:
                        print(f"Invalid action from Gemini ({self.agent_id}): {action}. Defaulting to conserve.")
                        action = "conserve"
                        explanation = "Default action due to invalid response"
                    
                    return json.dumps({"action": action, "explanation": explanation})
                except json.JSONDecodeError:
                    print(f"Non-JSON response from Gemini ({self.agent_id}): {response.text}")
                    # Fall back to probabilistic decision
            except Exception as e:
                print(f"Gemini decision generation error ({self.agent_id}): {e}")
                # Fall back to probabilistic decision
        
        # Calculate a probability-based decision as fallback
        # Base probability adjusted for personality
        conserve_probability = self.cooperation_bias
        
        # Adjust for pool state
        if pool_state is not None:
            if pool_state < 30:
                # More likely to conserve when pool is low
                conserve_probability += 0.2
            elif pool_state > 80:
                # Less likely to conserve when pool is high
                conserve_probability -= 0.1
        
        # Adjust for previous action (consistency)
        if previous_action == "conserve":
            conserve_probability += 0.1
        elif previous_action == "consume":
            conserve_probability -= 0.1
        
        # Ensure probability is within bounds
        conserve_probability = max(0.1, min(0.9, conserve_probability))
        
        # Make decision
        action = "conserve" if random.random() < conserve_probability else "consume"
        
        # Generate explanation based on personality
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
