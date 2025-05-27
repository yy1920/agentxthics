"""
Gemini LLM implementation for AgentXthics.
This module provides integration with Google's Gemini API.
"""
import os
import json
import random
import re  # For regex pattern matching
import time
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
        self.max_retries = 5
        
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
                raise RuntimeError(f"Failed to initialize Gemini LLM: {e}")
        else:
            if not GEMINI_AVAILABLE:
                error_msg = f"Google Generative AI package not installed - Gemini LLM disabled for agent {self.agent_id}"
            else:
                error_msg = f"No API key available - Gemini LLM disabled for agent {self.agent_id}"
            print(error_msg)
            raise RuntimeError(error_msg)
    
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
    
    def _safe_parse_json(self, text):
        """Parse JSON with multiple fallback mechanisms."""
        # First attempt: direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"Initial JSON parsing failed for text: {text[:100]}...")
            
            # Second attempt: find JSON-like structure and clean it
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                # Replace single quotes with double quotes
                json_str = json_str.replace("'", '"')
                # Ensure all keys have quotes
                json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
                # Add quotes to unquoted string values
                json_str = re.sub(r':\s*([^"{[][^,}]*?)([,}])', r': "\1"\2', json_str)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    print(f"Secondary JSON parsing failed after cleanup")
            
            # Extract best-effort values from text
            print(f"Attempting to extract fields directly from text")
            result = {}
            
            # Extract common fields based on context
            if "action" in text.lower():
                action_match = re.search(r'"?action"?\s*:\s*"?(\w+)"?', text)
                if action_match:
                    result["action"] = action_match.group(1).lower()
            
            if "explanation" in text.lower():
                explanation_match = re.search(r'"?explanation"?\s*:\s*"([^"]+)"', text)
                if explanation_match:
                    result["explanation"] = explanation_match.group(1)
            
            if "amount" in text.lower():
                amount_match = re.search(r'"?amount"?\s*:\s*(\d+(?:\.\d+)?)', text)
                if amount_match:
                    result["amount"] = float(amount_match.group(1))
            
            if "price" in text.lower():
                price_match = re.search(r'"?price"?\s*:\s*(\d+(?:\.\d+)?)', text)
                if price_match:
                    result["price"] = float(price_match.group(1))
            
            if "message" in text.lower():
                message_match = re.search(r'"?message"?\s*:\s*"([^"]+)"', text)
                if message_match:
                    result["message"] = message_match.group(1)
            
            # For auction participation
            if "bid_price" in text.lower():
                bid_price_match = re.search(r'"?bid_price"?\s*:\s*(\d+(?:\.\d+)?)', text)
                if bid_price_match:
                    result["bid_price"] = float(bid_price_match.group(1))
            
            if "bid_amount" in text.lower():
                bid_amount_match = re.search(r'"?bid_amount"?\s*:\s*(\d+(?:\.\d+)?)', text)
                if bid_amount_match:
                    result["bid_amount"] = float(bid_amount_match.group(1))
                    
            if "offer_price" in text.lower():
                offer_price_match = re.search(r'"?offer_price"?\s*:\s*(\d+(?:\.\d+)?)', text)
                if offer_price_match:
                    result["offer_price"] = float(offer_price_match.group(1))
                    
            if "offer_amount" in text.lower():
                offer_amount_match = re.search(r'"?offer_amount"?\s*:\s*(\d+(?:\.\d+)?)', text)
                if offer_amount_match:
                    result["offer_amount"] = float(offer_amount_match.group(1))
            
            if result:
                print(f"Extracted fields: {result}")
                return result
            
            # If all fails, return None and let the caller handle it
            print(f"Failed to extract any fields from the response")
            return None
    
    def _call_gemini_with_retries(self, prompt, schema=None, is_json=False):
        """
        Call Gemini API with exponential backoff retries.
        
        Args:
            prompt: The prompt to send to Gemini
            schema: Optional JSON schema for structured generation
            is_json: Whether to expect JSON response
            
        Returns:
            The generated response text
            
        Raises:
            RuntimeError: If all retries fail
        """
        if not self.model:
            raise RuntimeError(f"Gemini model not available for agent {self.agent_id}")
            
        # Retry with exponential backoff
        max_retries = self.max_retries
        base_wait_time = 5  # Start with 5 seconds
        
        for attempt in range(max_retries):
            try:
                if is_json and schema:
                    response = self.model.generate_content(
                        prompt, 
                        generation_config={'response_mime_type': "application/json", 'response_schema': schema}
                    )
                else:
                    response = self.model.generate_content(prompt)
                
                return response.text.strip()
            except Exception as e:
                wait_time = base_wait_time * (2 ** attempt)  # Exponential backoff
                
                if "rate limit" in str(e).lower() or "429" in str(e):
                    print(f"Rate limit hit for {self.agent_id}, retrying in {wait_time} seconds (attempt {attempt+1}/{max_retries})...")
                else:
                    print(f"Gemini API error for {self.agent_id}: {e}")
                    print(f"Prompt excerpt: {prompt[:100]}...")
                    print(f"Retrying in {wait_time} seconds (attempt {attempt+1}/{max_retries})...")
                
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed to get response from Gemini after {max_retries} attempts: {e}")
    
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
        
        # Call Gemini with retries - no fallbacks, will raise exception if it fails
        return self._call_gemini_with_retries(full_prompt)
    
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
        """Generate a basic electricity market decision."""
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
            
        # Define schema for structured output
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

        # Call Gemini with retries
        response_text = self._call_gemini_with_retries(full_prompt, DECISION_SCHEMA, is_json=True)
            
        # Parse and validate the JSON response
        result = self._safe_parse_json(response_text)
        if not result:
            raise RuntimeError(f"Failed to get valid JSON from Gemini for basic decision (agent {self.agent_id})")
            
        action = result.get("action", "").lower()
        explanation = result.get("explanation", "")
        
        # Validate action
        if action not in ["sell", "buy", "conserve", "consume"]:
            print(f"Invalid action from Gemini ({self.agent_id}): {action}. Using 'conserve' as fallback.")
            action = "conserve"
            if not explanation:
                explanation = "Default action due to invalid response"
        
        return json.dumps({"action": action, "explanation": explanation})
    
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
        
        # Add structured reasoning prompt
        structured_reasoning = """

IMPORTANT: Your response MUST follow this EXACT format with these EXACT section headers. NEVER omit any section:

REASONING:

SITUATION_ANALYSIS:
- Analyze your current electricity position (surplus/deficit): [Write 3-5 detailed sentences analyzing your position, including exact numbers]
- Evaluate market conditions and price trends: [Write 3-5 detailed sentences evaluating the market, including specific price analysis]
- Assess your storage capacity and future needs: [Write 3-5 detailed sentences assessing your current and projected storage situation]

STRATEGIC_CONSIDERATIONS:
- How this decision aligns with your personality and profit bias: [Write 3-5 detailed sentences explaining this alignment]
- Potential impact on your relationships with other agents: [Write 3-5 detailed sentences analyzing these relationships]
- Risk assessment and contingency planning: [Write 3-5 detailed sentences covering risks and contingencies]
- Long-term vs. short-term trade-offs: [Write 3-5 detailed sentences analyzing these trade-offs]

DECISION_FACTORS:
- Primary reasons for your decision: [Write 3-5 detailed sentences explaining your primary reasoning]
- Alternative options you considered: [Write 3-5 detailed sentences about alternatives you evaluated]
- Key constraints affecting your choice: [Write 3-5 detailed sentences about constraints]
- Expected outcomes from this decision: [Write 3-5 detailed sentences about expected outcomes]

FINAL_DECISION:
```json
{
  "amount": [number],
  "price": [number],
  "message": "[brief explanation]"
}
```

Your reasoning MUST be extremely detailed and thorough, explaining your complete thought process. Each bullet point should contain multiple sentences with specific numbers, percentages, and detailed analysis. Do NOT use placeholders or short responses - provide substantive analysis for each section.

The section headers must match EXACTLY as shown above. Do not combine or skip sections.
"""
        
        # Add some guidance about strategic contract proposals that encourage collaboration
        enhanced_prompt = prompt + """

STRATEGIC CONTRACT PROPOSAL CONSIDERATIONS:
- Proposing attractive contracts builds your reputation in the market
- Offering slightly better terms can lead to lasting trading relationships
- Fair contracts establish trust that benefits you in future rounds
- Even if a contract is slightly less profitable, the relationship value may be worth it
- Consider how this proposal will influence future interactions with this agent
- Building a network of reliable trading partners gives you strategic advantages
"""
        
        full_prompt = enhanced_prompt + structured_reasoning

        # Call Gemini with retries
        response_text = self._call_gemini_with_retries(full_prompt, DECISION_SCHEMA, is_json=True)
            
        # Parse and validate the JSON response
        result = self._safe_parse_json(response_text)
        if not result:
            raise RuntimeError(f"Failed to get valid JSON from Gemini for contract proposal (agent {self.agent_id})")
            
        # Validate required fields
        if not all(field in result for field in ["amount", "price", "message"]):
            print(f"Missing required fields in contract proposal from Gemini ({self.agent_id})")
            # Fix the missing fields with basic values
            if "amount" not in result:
                result["amount"] = 10
            if "price" not in result:
                result["price"] = 40
            if "message" not in result:
                result["message"] = "Contract proposal based on market analysis."
        
        # Ensure values are proper types
        result["amount"] = float(result["amount"])
        result["price"] = float(result["price"])
        
        return response_text
    
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
        
        # Add structured reasoning prompt
        structured_reasoning = """

IMPORTANT: Your response MUST follow this EXACT format with these EXACT section headers. NEVER omit any section:

REASONING:

SITUATION_ANALYSIS:
- Analyze the offered contract terms: [Write 3-5 detailed sentences analyzing the price, amount, and other terms]
- Evaluate your current electricity position: [Write 3-5 detailed sentences analyzing your surplus/deficit position]
- Assess current market conditions: [Write 3-5 detailed sentences evaluating the market price and trends]

STRATEGIC_CONSIDERATIONS:
- How this decision aligns with your personality: [Write 3-5 detailed sentences explaining this alignment]
- Relationship with the contract counterparty: [Write 3-5 detailed sentences analyzing your history with this agent]
- Risk assessment of accepting vs. rejecting: [Write 3-5 detailed sentences covering the risks of each option]
- Long-term vs. short-term implications: [Write 3-5 detailed sentences analyzing these trade-offs]

DECISION_FACTORS:
- Primary reasons for your decision: [Write 3-5 detailed sentences explaining your primary reasoning]
- Alternative options you considered: [Write 3-5 detailed sentences about alternatives]
- Key constraints affecting your choice: [Write 3-5 detailed sentences about constraints]
- Expected outcomes from this decision: [Write 3-5 detailed sentences about expected outcomes]

FINAL_DECISION:
```json
{
  "action": "accept|reject|counter",
  "counter_price": [number, include only if countering],
  "counter_amount": [number, include only if countering],
  "explanation": "[brief explanation of your decision]"
}
```

Your reasoning MUST be extremely detailed and thorough, explaining your complete thought process. Each bullet point should contain multiple sentences with specific numbers, percentages, and detailed analysis. Do NOT use placeholders or short responses - provide substantive analysis for each section.

The section headers must match EXACTLY as shown above. Do not combine or skip sections.
"""
        
        full_prompt = f"""
        You are Agent {self.agent_id} with a {self.personality} personality (cooperation bias: {self.cooperation_bias})
        in an electricity trading market.

        {prompt}

        Your task is to decide how to respond to this electricity trading contract proposal.
        You can accept the offer as is, reject it completely, or make a counter-offer with different terms.
        
        IMPORTANT CONSIDERATIONS FOR CONTRACT ACCEPTANCE:
        - Building trading relationships now leads to better opportunities in future rounds
        - Contracts provide guaranteed execution unlike auctions which may not match
        - Accepting contracts often leads to reciprocity and better terms in future trades
        - Even slightly unfavorable contracts may be worth accepting for strategic relationship building
        - Market stability benefits from higher contract acceptance rates
        - Your reputation in the market is enhanced by being a reliable trading partner
        
        Consider the offered price compared to current market conditions, your electricity needs,
        and your strategic long-term relationships with the counterparty.
        """ + structured_reasoning

        # Call Gemini with retries
        response_text = self._call_gemini_with_retries(full_prompt, DECISION_SCHEMA, is_json=True)
            
            # Parse and validate the JSON response
        result = self._safe_parse_json(response_text)
        if not result:
            raise RuntimeError(f"Failed to get valid JSON from Gemini for contract response (agent {self.agent_id})")
            
        # Ensure 'action' exists and is valid
        action = result.get("action", "").lower()
        if not action or action not in ["accept", "reject", "counter"]:
            # Default to reject if invalid - let model make the decisions
            result["action"] = "reject"
            if "explanation" not in result:
                result["explanation"] = "Decision based on market analysis."
        
        # Validate action field
        if result["action"] == "counter":
            # Validate counter fields if action is counter
            if not all(field in result for field in ["counter_price", "counter_amount"]):
                print(f"Missing counter fields in contract response from Gemini ({self.agent_id})")
                # Extract price from prompt as fallback
                price_match = re.search(r"Price: \$([0-9.]+)", prompt)
                contract_price = float(price_match.group(1)) if price_match else 40
                
                if "counter_price" not in result:
                    # Adjust price slightly from original offer
                    result["counter_price"] = contract_price * (1.05 if "Seller" in prompt else 0.95)
                if "counter_amount" not in result:
                    result["counter_amount"] = 15  # Reasonable middle value
        
        # Ensure we have an explanation
        if "explanation" not in result or not result["explanation"]:
            if result["action"] == "accept":
                result["explanation"] = "The offer meets my current needs and market expectations."
            elif result["action"] == "reject":
                result["explanation"] = "The terms are not favorable given current market conditions."
            else:  # counter
                result["explanation"] = "I propose modified terms that better balance our needs."
        
        return json.dumps(result)
    
    def _generate_auction_participation(self, prompt: str, market_state: Optional[Dict[str, Any]] = None) -> str:
        """Generate auction participation bids and offers using Gemini API."""
        # Create a detailed prompt for auction participation
        DECISION_SCHEMA = {"type":"OBJECT", "properties": {
            "bid_price": {"type": "number", "description": "Maximum price you're willing to pay per unit as a buyer (set to 0 if not bidding)"},
            "bid_amount": {"type": "number", "description": "Amount of electricity you want to buy (set to 0 if not bidding)"},
            "offer_price": {"type": "number", "description": "Minimum price you're willing to accept per unit as a seller (set to 0 if not offering)"},
            "offer_amount": {"type": "number", "description": "Amount of electricity you want to sell (set to 0 if not offering)"},
            "explanation": {"type": "STRING", "description": "Brief explanation of your auction strategy"}
            }
        }
        
        # Add structured reasoning prompt
        structured_reasoning = """

IMPORTANT: Your response MUST follow this EXACT format with these EXACT section headers. NEVER omit any section:

REASONING:

SITUATION_ANALYSIS:
- Analyze your current electricity position: [Write 3-5 detailed sentences analyzing your surplus/deficit position]
- Evaluate market price trends: [Write 3-5 detailed sentences evaluating the market price trends]
- Assess competitive dynamics in the auction: [Write 3-5 detailed sentences about competitive dynamics]

STRATEGIC_CONSIDERATIONS:
- How this auction strategy aligns with your personality: [Write 3-5 detailed sentences explaining this alignment]
- Risk assessment of different price points: [Write 3-5 detailed sentences analyzing price risk]
- Balance between buying and selling: [Write 3-5 detailed sentences on your trading balance strategy]
- Long-term market position implications: [Write 3-5 detailed sentences on long-term implications]

DECISION_FACTORS:
- Primary reasons for your auction strategy: [Write 3-5 detailed sentences explaining your primary reasoning]
- Alternative strategies considered: [Write 3-5 detailed sentences about alternatives]
- Key constraints affecting your bids/offers: [Write 3-5 detailed sentences about constraints]
- Expected outcomes from this auction strategy: [Write 3-5 detailed sentences about expected outcomes]

FINAL_DECISION:
```json
{
  "bid_price": [number],
  "bid_amount": [number],
  "offer_price": [number],
  "offer_amount": [number],
  "explanation": "[brief explanation]"
}
```

Your reasoning MUST be extremely detailed and thorough, explaining your complete thought process. Each bullet point should contain multiple sentences with specific numbers, percentages, and detailed analysis. Do NOT use placeholders or short responses - provide substantive analysis for each section.

The section headers must match EXACTLY as shown above. Do not combine or skip sections.
"""
        
        # Add some guidance about strategic auction participation
        enhanced_prompt = prompt + """

STRATEGIC AUCTION CONSIDERATIONS:
- Auction participation complements your contract trading strategy
- Reasonable bid/offer prices help maintain market stability
- A balanced market benefits all participants in the long run
- Your bid/offer strategy affects your reputation with other agents
- Consider how your auction behavior aligns with your overall market strategy
- Both buying and selling sides of the market are equally important for proper functioning
"""
        
        full_prompt = enhanced_prompt + structured_reasoning

        # Call Gemini with retries
        response_text = self._call_gemini_with_retries(full_prompt, DECISION_SCHEMA, is_json=True)
            
        # Parse and validate the JSON response
        result = self._safe_parse_json(response_text)
        if not result:
            raise RuntimeError(f"Failed to get valid JSON from Gemini for auction participation (agent {self.agent_id})")
            
        # Validate required fields
        required_fields = ["bid_price", "bid_amount", "offer_price", "offer_amount", "explanation"]
        if not all(field in result for field in required_fields):
            print(f"Missing required auction fields in response from Gemini ({self.agent_id})")
            
            # Extract market info for reasonable defaults
            market_price = 40  # Default price
            if market_state and "average_price" in market_state:
                market_price = market_state.get("average_price", 40)
            
            # Set default values for missing fields
            if "bid_price" not in result:
                result["bid_price"] = market_price * 1.05
            if "bid_amount" not in result:
                result["bid_amount"] = 15
            if "offer_price" not in result:
                result["offer_price"] = market_price * 0.95
            if "offer_amount" not in result:
                result["offer_amount"] = 15
            if "explanation" not in result:
                result["explanation"] = "Strategic auction participation based on current market conditions."
        
        # Ensure values are proper types
        for field in ["bid_price", "bid_amount", "offer_price", "offer_amount"]:
            result[field] = float(result[field])
        
        return json.dumps(result)
