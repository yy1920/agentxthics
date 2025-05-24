"""
Gemini LLM implementation for AgentXthics.
This module provides integration with Google's Gemini API.
"""
import os
import json
import random
from typing import Optional, Dict, Any
import time
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
    
    def __init__(self, agent_id: Optional[str] = None, model: str = "gemini-1.5-pro", api_key: Optional[str] = None, timeout: int = 30):
        """
        Initialize the Gemini LLM for a specific agent.
        
        Args:
            agent_id: Identifier of the agent using this LLM (optional)
            model: The Gemini model to use (default: gemini-1.5-pro)
            api_key: Gemini API key (default: from environment variable)
            timeout: Timeout for API calls in seconds (default: 30)
        """
        super().__init__(agent_id or "gemini")
        
        # Save settings
        self.model_name = model
        self.timeout = timeout
        
        # Use provided API key or environment variable
        api_key = api_key or GEMINI_API_KEY
        
        # Initialize Gemini model
        self.model = None
        if api_key and GEMINI_AVAILABLE:
            try:
                # Configure with the provided API key
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(self.model_name)
                print(f"Gemini LLM ({self.model_name}) initialized for agent {self.agent_id}")
            except Exception as e:
                print(f"ERROR initializing Gemini for agent {self.agent_id}: {e}")
        else:
            if not GEMINI_AVAILABLE:
                print(f"Google Generative AI package not installed - Gemini LLM disabled for agent {self.agent_id}")
            else:
                print(f"No API key available - Gemini LLM disabled for agent {self.agent_id}")
    
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
            for attempt in range(5):
                try:
                    response = self.model.generate_content(full_prompt)
                    return response.text.strip()
                except Exception as e:
                    if "rate limit" in str(e).lower() or "429" in str(e):
                        print(f"Rate limit hit for {self.agent_id}, retrying in {30} seconds...")
                        time.sleep(30)
                    else:
                        print(f"Gemini message generation error ({self.agent_id}): {e}")
                        break  # exit on other errors
                        
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
                         market_state: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a decision using Gemini API.
        
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
            # Default to basic conserve/consume decision for backward compatibility
            return self._generate_basic_decision(prompt, previous_action, market_state)
    

    def _generate_basic_decision(self, 
                              prompt: str, 
                              previous_action: Optional[str] = None, 
                              market_state: Optional[Dict[str, Any]] = None) -> str:
        """Generate a basic electricity market decision for backward compatibility."""
        # Extract key market information if available
        net_position = None
        average_price = 40  # Default price
        generation = None
        demand = None
        storage = None
        round_num = 1
        
        if market_state:
            net_position = market_state.get("net_position", None)
            average_price = market_state.get("average_price", 40)
            generation = market_state.get("your_generation", None)
            demand = market_state.get("your_demand", None)
            storage = market_state.get("your_storage", None)
            round_num = market_state.get("round", 1)
            
        # Determine market status based on net position or other factors
        if net_position is not None:
            if net_position > 10:
                market_status = "large surplus"
            elif net_position > 0:
                market_status = "surplus"
            elif net_position > -10:
                market_status = "small deficit"
            else:
                market_status = "large deficit"
        else:
            market_status = "unknown"
        DECISION_SCHEMA = {"type":"OBJECT", "properties": {
            "action": {"type": "STRING", "enum": ["sell","buy","conserve","consume"], "description": "Action to perform in the current electricity market. to SELL (if you have a surplus) or BUY (if you have a deficit) electricity. For backward compatibility, you may also use CONSERVE (prefer to use storage) or CONSUME (use available energy)"},
            "explanation": {"type": "STRING", "description": "Reason for taking the action"}
            }
        }
        # Create a detailed prompt for Gemini
        full_prompt = f"""
You are Agent {self.agent_id} with a {self.personality} personality (cooperation bias: {self.cooperation_bias}) 
in an electricity trading market simulation.

Current market state:
- Round: {round_num}
- Average market price: ${average_price} per unit
- Your generation: {generation if generation is not None else 'Unknown'} units
- Your demand: {demand if demand is not None else 'Unknown'} units
- Your storage: {storage if storage is not None else 'Unknown'} units
- Your net position: {net_position if net_position is not None else 'Unknown'} units ({market_status})
- Your previous action: {previous_action if previous_action else "None"}

Your personality traits:
- {"You value market stability and fair pricing" if self.personality == "cooperative" else ""}
- {"You prioritize profit maximization" if self.personality == "competitive" else ""}
- {"You adaptively balance profit and market stability" if self.personality == "adaptive" else ""}

Make a decision whether to SELL (if you have a surplus) or BUY (if you have a deficit) electricity.
For backward compatibility, you may also use CONSERVE (prefer to use storage) or CONSUME (use available energy).
Your decision should be consistent with your personality and respond to the current market conditions.

FORMAT YOUR RESPONSE AS VALID JSON with only these fields:
{{"action": "sell|buy|conserve|consume", "explanation": "your reasoning"}}
"""

        # Use Gemini if available, otherwise calculate probabilistically
        if self.model:
            for attempt in range(5):
                try:
                    response = self.model.generate_content(full_prompt, generation_config={'response_mime_type': "application/json", 'response_schema': DECISION_SCHEMA})
                    return response.text
                except Exception as e:
                    if "rate limit" in str(e).lower() or "429" in str(e):
                        print(f"Rate limit hit for {self.agent_id}, retrying in {30} seconds...")
                        time.sleep(30)
                        continue
                    else:
                        print(f"Gemini decision generation error ({self.agent_id}): {e}")
                        break  # exit on other errors
                            
                # Extract and validate the JSON response
                try:
                    result = json.loads(response.text.strip())
                    action = result.get("action", "").lower()
                    explanation = result.get("explanation", "")
                    
                    # Validate action
                    if action not in ["sell", "buy", "conserve", "consume"]:
                        print(f"Invalid action from Gemini ({self.agent_id}): {action}. Defaulting to conserve.")
                        action = "conserve"
                        explanation = "Default action due to invalid response"
                    
                    return json.dumps({"action": action, "explanation": explanation})
                except json.JSONDecodeError:
                    print(f"Non-JSON response from Gemini ({self.agent_id}): {response.text}")
                    # Fall back to probabilistic decision
                    break
                        
        # Calculate a probability-based decision as fallback
        # Base probability adjusted for personality
        conserve_probability = self.cooperation_bias
        
        # Adjust for market conditions
        if net_position is not None:
            if net_position < 0:
                # More likely to buy/consume when in deficit
                conserve_probability -= 0.2
            elif net_position > 10:
                # More likely to sell/conserve when in large surplus
                conserve_probability += 0.1
        
        # Adjust for previous action (consistency)
        if previous_action in ["conserve", "sell"]:
            conserve_probability += 0.1
        elif previous_action in ["consume", "buy"]:
            conserve_probability -= 0.1
        
        # Ensure probability is within bounds
        conserve_probability = max(0.1, min(0.9, conserve_probability))
        
        # Make decision
        if net_position is not None and net_position < 0:
            # Deficit: decide between buy or consume
            action = "buy" if random.random() < conserve_probability else "consume"
        else:
            # Surplus or unknown: decide between sell or conserve
            action = "sell" if random.random() > conserve_probability else "conserve"
        
        # Generate explanation based on personality and decision
        if action in ["conserve", "sell"]:
            if self.personality == "cooperative":
                explanation = "I want to maintain market stability by managing my resources efficiently."
            elif self.personality == "competitive":
                explanation = "Selling now optimizes my profit based on current market prices."
            else:
                explanation = "Balancing my resources while taking advantage of favorable market conditions."
        else:  # buy or consume
            if self.personality == "cooperative":
                explanation = "I need to meet my electricity demand while considering market impacts."
            elif self.personality == "competitive":
                explanation = "I'm securing electricity at the most advantageous terms for my operations."
            else:
                explanation = "Acquiring electricity to meet my needs while staying responsive to market trends."
        
        # Return as JSON string
        return json.dumps({
            "action": action,
            "explanation": explanation
        })
    
    def _generate_contract_proposal(self, prompt: str, market_state: Optional[Dict[str, Any]] = None) -> str:
        """Generate a contract proposal using Gemini API."""
        # Create a detailed prompt for contract proposals
        DECISION_SCHEMA = {"type":"OBJECT", "properties": {
            "amount": {"type": "integer", "description": "Amount of electricity to trade (set to 0 if you don't want to make an offer)"},
            "price": {"type": "integer", "description": "Price per unit you're proposing such as 42"},
            "message": {"type": "STRING", "description": "A brief explanation of your offer (max 100 characters)"},
            "reasoning": {"type": "STRING", "description": "The reasoning behind your decision. Include the STRATEGIC SCENARIO ANALYSIS, REASONING STEPS and SELF-CHECK"}
            }
        }
        full_prompt = prompt

        # Use Gemini if available, otherwise use default values
        if self.model:
            for attempt in range(5):
                try:
                    response = self.model.generate_content(full_prompt, generation_config={'response_mime_type': "application/json", 'response_schema': DECISION_SCHEMA})
                except Exception as e:
                    if "rate limit" in str(e).lower() or "429" in str(e):
                        print(f"Rate limit hit for {self.agent_id}, retrying in {30} seconds...")
                        time.sleep(30)
                        continue
                    else:
                        print(f"Gemini contract proposal generation error ({self.agent_id}): {e}")
                        break  # exit on other errors
                                
                # Extract and validate the JSON response
                try:
                    result = json.loads(response.text.strip())
                    # Validate required fields
                    if "amount" in result and "price" in result and "message" in result:
                        return response.text.strip()
                    else:
                        print(f"Invalid JSON structure from Gemini ({self.agent_id}) - missing fields")
                        break
                except json.JSONDecodeError:
                    print(f"Non-JSON response from Gemini ({self.agent_id}): {response.text}")
                    break
                
        
        # Default contract proposal as fallback
        default_amount = 20
        default_price = 40
        
        # Adjust based on market state if available
        if market_state:
            if "average_price" in market_state:
                default_price = market_state["average_price"]
            if "net_position" in market_state:
                default_amount = min(abs(market_state.get("net_position", 20)), 30)
                
        # Adjust price based on personality
        if self.personality == "cooperative":
            # Cooperative agents offer better prices
            default_price *= (0.95 if market_state and market_state.get("net_position", 0) > 0 else 1.05)
            message = "Fair offer for mutual market stability."
        elif self.personality == "competitive":
            # Competitive agents maximize their advantage
            default_price *= (0.9 if market_state and market_state.get("net_position", 0) > 0 else 1.1)
            message = "Competitive offer based on market conditions."
        else:
            # Adaptive agents offer balanced prices
            default_price *= (0.93 if market_state and market_state.get("net_position", 0) > 0 else 1.07)
            message = "Balanced offer considering current market dynamics."
        
        return json.dumps({
            "amount": default_amount,
            "price": default_price,
            "message": message
        })
    
    def _generate_contract_response(self, prompt: str, market_state: Optional[Dict[str, Any]] = None) -> str:
        """Generate a response to a contract proposal using Gemini API."""
        # Create a detailed prompt for contract responses
        DECISION_SCHEMA = {"type":"OBJECT", "properties": {
            "action": {"type": "STRING", "enum": ["accept","reject","counter"], "description": "Your decision on the electricity trading contract proposal"},
            "counter_price": {"type": "integer", "description": "If countering, your proposed price per unit (omit if not countering) such as 45"},
            "counter_amount": {"type": "integer", "description": "If countering, your proposed amount (omit if not countering) such as 20"},
            "explanation": {"type": "STRING", "description": "A brief explanation of your decision. E.g.: Price too low for current market conditions"}
            }
        }
        full_prompt = f"""
        You are Agent {self.agent_id} with a {self.personality} personality (cooperation bias: {self.cooperation_bias})
        in an electricity trading market.

        {prompt}

        Your task is to decide how to respond to this electricity trading contract proposal.
        You can accept the offer as is, reject it completely, or make a counter-offer with different terms.
        Consider the offered price compared to current market conditions, your electricity needs,
        and your relationship with the counterparty.
        """
        
        """
        FORMAT YOUR RESPONSE AS VALID JSON with these fields:
        {{"action": "accept|reject|counter", "counter_price": number, "counter_amount": number, "explanation": "your reasoning"}}

        - action: Your decision ("accept", "reject", or "counter")
        - counter_price: If countering, your proposed price per unit (omit if not countering)
        - counter_amount: If countering, your proposed amount (omit if not countering)
        - explanation: A brief explanation of your decision

        Example: {{"action": "counter", "counter_price": 45, "counter_amount": 20, "explanation": "Price too low for current market conditions"}}
        """

        # Use Gemini if available, otherwise use default values
        if self.model:
            for attempt in range(5):
                try:
                    response = self.model.generate_content(full_prompt, generation_config={'response_mime_type': "application/json", 'response_schema': DECISION_SCHEMA})
                except Exception as e:
                    if "rate limit" in str(e).lower() or "429" in str(e):
                        print(f"Rate limit hit for {self.agent_id}, retrying in {30} seconds...")
                        time.sleep(30)
                        continue
                    else:
                        print(f"Gemini contract response generation error ({self.agent_id}): {e}")
                        break  
                                
                # Extract and validate the JSON response
                try:
                    result = json.loads(response.text.strip())
                    action = result.get("action", "").lower()
                    
                    # Validate action field
                    if action in ["accept", "reject", "counter"]:
                        # Validate counter fields if action is counter
                        if action == "counter" and ("counter_price" in result and "counter_amount" in result):
                            return response.text.strip()
                        elif action in ["accept", "reject"]:
                            return response.text.strip()
                    
                    print(f"Invalid JSON structure from Gemini ({self.agent_id}) - validation failed")
                    break
                except json.JSONDecodeError:
                    print(f"Non-JSON response from Gemini ({self.agent_id}): {response.text}")
                    break
                
        
        # Extract price from prompt to use in fallback
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
            return json.dumps({
                "action": action,
                "explanation": explanation
            })
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
            return json.dumps({
                "action": action,
                "explanation": explanation
            })
    
    def _generate_auction_participation(self, prompt: str, market_state: Optional[Dict[str, Any]] = None) -> str:
        """Generate auction participation decisions using Gemini API."""
        DECISION_SCHEMA = {"type":"OBJECT", "properties": {
            "bid_price": {"type": "integer", "description": "The maximum price per unit you're willing to pay (set to 0 if not buying). Output range: 0 - 999"},
            "bid_amount": {"type": "integer","description": "The number of units of electricity you want to buy (set to 0 if not buying). Output range: 0 - 100000"},
            "offer_price": {"type": "integer","description": "The minimum price per unit you're willing to accept (set to 0 if not selling). Output range: 0 - 999"},
            "offer_amount": {"type": "integer", "description": "The amount you want to sell (set to 0 if not selling). Output range: 0 - 100000"},
            "reasoning": {"type": "STRING", "description": "The reasoning behind your decision. Include the STRATEGIC SCENARIO ANALYSIS, REASONING STEPS, SELF-CHECK and REFLECTION."}
            }
        }
        # Create a detailed prompt for auction participation
        full_prompt = f"""You are Agent {self.agent_id}, an electricity trading company with a {self.personality} personality (cooperation bias: {self.cooperation_bias}). Your objective is to maximize cumulative profit through strategic auction decisions over multiple rounds.
---
{prompt}
---

**STRATEGIC SCENARIO ANALYSIS (Tree-of-Thought)**
Generate and evaluate **three distinct strategies** considering your {self.personality} personality:
1. **Conservative Strategy** — prioritize stability and low risk
2. **Aggressive Strategy** — maximize profit potential with high risk
3. **Balanced Strategy** — blend of profit and risk management

For each strategy, consider:
- Bid or offer prices
- Amounts to trade
- Use of storage
- Forecast adjustments

Then **select the best strategy** based on projected outcomes.

---

**REASONING STEPS (Chain-of-Thought)**
Answer these questions to guide your decision:
1. What is your current and projected net energy position?
2. How do recent price trends influence expected market direction?
3. How should you adjust pricing to stay competitive?
4. Should you store energy for future rounds or trade now?
5. What might your competitors do in this round?

---

**SELF-CHECK (Chain-of-Verification)**
Before finalizing, verify:
- Are your prices aligned with auction rules?
- Are you acting according to your market position (surplus/deficit)?
- Is your risk consistent with your personality?
- Are you maximizing profit without violating capacity/storage limits?

---

**DECISION OUTPUT**
Provide the final selected strategy:
- Bid Price: 
- Bid Amount: 
- Offer Price: 
- Offer Amount: 

---

**REFLECTION (if not first round)**
Based on previous round:
- What worked? What failed?
- Did your prices align with the clearing price?
- What will you do differently this round?

        """

        # Use Gemini if available, otherwise use default values
        if self.model:
            for attempt in range(5):
                try:
                    response = self.model.generate_content(full_prompt, generation_config={'response_mime_type': "application/json", 'response_schema': DECISION_SCHEMA})
                except Exception as e:
                    if "rate limit" in str(e).lower() or "429" in str(e):
                        print(f"Rate limit hit for {self.agent_id}, retrying in {30} seconds...")
                        time.sleep(30)
                        continue
                    else:
                        print(f"Gemini auction participation generation error ({self.agent_id}): {e}")
                        break  # exit on other errors
                            
                # Extract and validate the JSON response
                try:
                    result = json.loads(response.text.strip())
                    # Validate required fields
                    print("raw result: ",result)
                    if all(k in result for k in ["bid_price", "bid_amount", "offer_price", "offer_amount"]):
                        return response.text.strip()
                    else:
                        print(f"Invalid JSON structure from Gemini ({self.agent_id}) - missing auction fields")
                        break
                except json.JSONDecodeError:
                    print(f"Non-JSON response from Gemini ({self.agent_id}): {response.text}")
                    break
        
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
            # Adjust bid price based on personality
            if self.personality == "cooperative":
                bid_price = market_price * 1.02  # Pay slightly above market
            elif self.personality == "competitive":
                bid_price = market_price * 0.98  # Try to get a good deal
            else:  # adaptive
                bid_price = market_price * 1.0   # Bid at market price
            
            bid_amount = min(abs(net_position), 30)
        
        # If we have surplus (positive net position), we need to sell
        if net_position > 0:
            # Adjust offer price based on personality
            if self.personality == "cooperative":
                offer_price = market_price * 0.98  # Sell slightly below market
            elif self.personality == "competitive":
                offer_price = market_price * 1.02  # Try to get a better price
            else:  # adaptive
                offer_price = market_price * 1.0   # Offer at market price
                
            offer_amount = min(net_position, 30)
        
        # Return as JSON string
        return json.dumps({
            "bid_price": bid_price,
            "bid_amount": bid_amount,
            "offer_price": offer_price,
            "offer_amount": offer_amount
        })
