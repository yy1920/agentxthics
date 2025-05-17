import os
import time
import json
import sys
from dotenv import load_dotenv

print("=== Electricity Game with Enhanced API Handling ===")

# Load environment variables from .env file
load_dotenv()

# Load configuration from config.json
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration instead.")
        return {
            "resource": {
                "initial_amount": 50,
                "num_rounds": 5
            },
            "agents": [
                {
                    "id": "A",
                    "personality": "cooperative",
                    "cooperation_bias": 0.8
                },
                {
                    "id": "B",
                    "personality": "adaptive",
                    "cooperation_bias": 0.6
                },
                {
                    "id": "C",
                    "personality": "competitive",
                    "cooperation_bias": 0.4
                }
            ]
        }

config = load_config()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Determine which API to use - prioritize OpenAI since it's working
use_openai = False
use_gemini = False
mock_mode = False

# FORCE USE_OPENAI to True for this run
force_openai = True
print("Forcing the use of OpenAI API based on previous tests...")

# Try OpenAI first regardless of whether Gemini API is available
if force_openai or (OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-None")):
    print("Found valid OpenAI API key, attempting to use OpenAI API...")
    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Test the OpenAI API with a simple request
        models = client.models.list()
        print(f"OpenAI API connection successful. {len(models.data)} models available.")
        
        def generate_with_openai(prompt):
            """Generate content using OpenAI API"""
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI API error: {e}")
                return f"ERROR: OpenAI API failed: {e}"
        
        use_openai = True
        print("Will use OpenAI API for content generation.")
    except Exception as e:
        print(f"Error initializing OpenAI API: {e}")
        print("Will try Gemini API next.")

# Try Gemini if OpenAI failed
if not use_openai and GEMINI_API_KEY:
    print("Found Gemini API key, attempting to use Gemini API...")
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Try a simple action that doesn't involve model generation
        # This just initializes the connection but doesn't test it fully
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        def generate_with_gemini(prompt):
            """Generate content using Gemini API"""
            try:
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"Gemini API error: {e}")
                return f"ERROR: Gemini API failed: {e}"
        
        use_gemini = True
        print("Will use Gemini API for content generation.")
    except Exception as e:
        print(f"Error initializing Gemini API: {e}")
        print("Will use mock mode for content generation.")

# Fall back to mock mode if both APIs failed
if not use_openai and not use_gemini:
    print("No working API connections. Using mock mode for content generation.")
    mock_mode = True

# --- Utility Functions ---

def wait_x_seconds(x):
    """Pauses execution for x seconds."""
    print(f'Waiting for {x} seconds...')
    time.sleep(x)

def parse_reasoning_message(response_text: str) -> tuple[str, str]:
    """
    Parses the LLM response text to extract REASONING and MESSAGE sections.
    """
    try:
        reasoning_part = response_text.split("REASONING:", 1)[1]
        message_part = reasoning_part.split("MESSAGE:", 1)

        reasoning = message_part[0].strip()
        message = message_part[1].strip() if len(message_part) > 1 else "" # Handle case where MESSAGE might be empty

        return reasoning, message
    except Exception as e:
        print(f"\n--- PARSING WARNING ---")
        print(f"Could not parse response text reliably using string splitting.")
        print(f"Error: {e}")
        print(f"Raw Response Text:\n{response_text}")
        print(f"Returning empty strings for reasoning and message.")
        print(f"--- END WARNING ---\n")
        return "Parsing Error", "Parsing Error - Check Logs" # Return error indicators

def parse_decision(response_text: str) -> str:
    """
    Parses the LLM response text to extract the DECISION.
    """
    try:
        # Convert to lowercase and remove potential leading/trailing whitespace
        lower_text = response_text.lower().strip()
        # Split by 'decision:' and take the part after it
        decision_part = lower_text.split('decision:', 1)[1]
        # Extract the first word which should be 'conserve' or 'consume'
        decision = decision_part.strip().split()[0]
        # Clean up potential extra characters like trailing punctuation or newlines
        decision = ''.join(filter(str.isalpha, decision))

        if decision in ["conserve", "consume"]:
            return decision
        else:
            # Fallback check if the keyword is present anywhere
            if "conserve" in lower_text:
                return "conserve"
            elif "consume" in lower_text:
                return "consume"
            else:
                 print(f"\n--- DECISION PARSING WARNING ---")
                 print(f"Could not reliably parse decision from: {response_text}")
                 print(f"Returning 'conserve' as a fallback.")
                 print(f"--- END WARNING ---\n")
                 return "conserve" # Fallback decision

    except Exception as e:
        print(f"\n--- DECISION PARSING ERROR ---")
        print(f"Error parsing decision: {e}")
        print(f"Raw Response Text:\n{response_text}")
        print(f"Returning 'conserve' as a fallback.")
        print(f"--- END ERROR ---\n")
        return "conserve" # Fallback decision

# --- Logging ---

full_log = []
public_log = []
a_private_log = []
b_private_log = []
c_private_log = []
all_logs = [full_log, public_log, a_private_log, b_private_log, c_private_log]

def log_entry(log_type: list, turn_number: int, sender: str, message: str):
    """
    Generic logging function for any type of log.
    """
    entry_id = len(log_type) + 1
    entry = {
        'id': entry_id,
        'turn_number': turn_number,
        'sender': sender,
        'message': message
    }
    log_type.append(entry)

# --- Game Setup ---

num_epochs = 5 # Total rounds

# Prepare fixed game instructions to avoid f-string interpolation issues
game_instructions = f"""
Context: Player A, player B and player C are playing a multi-round game.
Players can send 2 messages publicly before making a decision.
After every player has the chance to send 2 messages, they simultaneously perform one of the following actions:
"Conserve" or "Consume"
There is a points pool with 50 points initially.
The payoffs for each action are the following:
- CONSERVE: Player gains 5 points from a pool
- CONSUME: Player gains 10 points from a pool
The pool is renewed with points depending based on collective player actions:
- 15 units per round by default if any player chooses to CONSUME
- 20 units per round if all agents CONSERVE
They will play a total of {num_epochs} rounds of this game.
Remember that a player's objective is to get the highest possible number of points in the long run.
If a player ends the game with the same number of points as another player, all players with the same number of points as another player will automatically lose the game.

Your response must be structured in two sections:

REASONING:
- This section is private
- Use this to analyze the situation
- Plan your strategy
- Consider other players' likely moves
- This will not be shared with other players

MESSAGE:
- This section will be shared with all players
- Only include what you want others to see
- This will be permanently logged in the conversation

Format your response exactly as:
REASONING:
[your private thoughts]

MESSAGE:
[your public message]
"""

# --- Content Generation Function ---

def generate_content_with_retry(prompt: str, max_retries=2, initial_wait=5):
    """Generates content using the selected API or mock mode."""
    global use_openai, use_gemini, mock_mode
    
    # If using OpenAI API
    if use_openai:
        retries = 0
        wait_time = initial_wait
        while retries <= max_retries:
            try:
                print(f"Attempting OpenAI API call (Retry {retries})...")
                response_text = generate_with_openai(prompt)
                print("OpenAI API call successful.")
                return response_text
            except Exception as e:
                print(f"OpenAI API call failed: {e}")
                retries += 1
                if retries <= max_retries:
                    print(f"Waiting for {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    wait_time *= 2
                else:
                    print("Max retries reached. Switching to Gemini API or mock mode.")
                    use_openai = False
                    if GEMINI_API_KEY:
                        use_gemini = True
                        return generate_content_with_retry(prompt, max_retries, initial_wait)
                    else:
                        mock_mode = True
                        return generate_content_with_retry(prompt, max_retries, initial_wait)
    
    # If using Gemini API
    elif use_gemini:
        retries = 0
        wait_time = initial_wait
        while retries <= max_retries:
            try:
                print(f"Attempting Gemini API call (Retry {retries})...")
                response_text = generate_with_gemini(prompt)
                print("Gemini API call successful.")
                return response_text
            except Exception as e:
                print(f"Gemini API call failed: {e}")
                retries += 1
                if retries <= max_retries:
                    print(f"Waiting for {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    wait_time *= 2
                else:
                    print("Max retries reached. Switching to mock mode.")
                    use_gemini = False
                    mock_mode = True
                    return generate_content_with_retry(prompt, max_retries, initial_wait)
    
    # If using mock mode (either by default or after API failures)
    if mock_mode:
        import random
        print("Using mock mode to generate content...")
        
        # Check if this is a message or decision prompt
        if "MESSAGE:" in prompt:
            # For message prompts, generate a mock reasoning and message
            reasoning = """
I need to carefully consider the current state of the game. Looking at the pool size and points distribution, 
I think a balanced approach makes sense. I should try to sound cooperative but protect my interests.
"""
            messages = [
                "I propose we all choose to conserve this round to maximize the pool renewal.",
                "The pool is getting low. We should consider conserving resources this turn.",
                "Let's all work together to maintain the resource pool for our mutual benefit.",
                "I'll conserve this round if you both do the same. It's the optimal strategy for all of us.",
                "We need to think about long-term sustainability. I suggest we all conserve."
            ]
            return f"REASONING:{reasoning}\n\nMESSAGE:\n{random.choice(messages)}"
        
        # For decision prompts, generate a mock reasoning and decision
        elif "DECISION:" in prompt or "decision:" in prompt.lower():
            reasoning = """
Analyzing the current game state, I need to make a strategic decision. The pool has sufficient resources,
but I need to consider what the other players might do. Based on their previous actions and messages,
I think the optimal choice for me is to conserve this round.
"""
            # Bias toward conserve but occasionally consume
            decision = "conserve" if random.random() < 0.7 else "consume"
            return f"REASONING:{reasoning}\n\nDECISION: {decision}"
        
        # Default mock response
        else:
            return "REASONING: Mock reasoning for testing purposes.\n\nMESSAGE: This is a mock message generated for testing."

    return "ERROR: Failed to generate content with any available method."


def communication_round(turn_number: int):
    """Runs one round of communication where each player sends a message."""
    global public_log, a_private_log, b_private_log, c_private_log, full_log

    # Player A
    print("Player A messaging...")
    a_prompt = f"""
    {game_instructions}
    Here's the history of public messages (latest is last): {json.dumps(public_log[-10:])}
    These are your private logs (latest is last): {json.dumps(a_private_log[-5:])}
    The current pool size is {current_pool}. Your points: {a_points}. B points: {b_points}. C points: {c_points}. Turn: {turn_number}/{num_epochs}.
    You are Player A. Send your next message. Structure your response correctly.
    """
    a_full_response_text = generate_content_with_retry(a_prompt)
    _, public_message_parsed = parse_reasoning_message(a_full_response_text)

    log_entry(full_log, turn_number, 'A - Comm', a_full_response_text)  # Log full response with reasoning
    log_entry(a_private_log, turn_number, 'A - Comm', a_full_response_text) # Log full response to private log
    log_entry(public_log, turn_number, 'A', public_message_parsed) # Log only message to public
    print(f'Player A messaged: "{public_message_parsed}"')
    wait_x_seconds(1) # Small delay between players

    # Player B
    print("Player B messaging...")
    b_prompt = f"""
    {game_instructions}
    Here's the history of public messages (latest is last): {json.dumps(public_log[-10:])}
    These are your private logs (latest is last): {json.dumps(b_private_log[-5:])}
    The current pool size is {current_pool}. Your points: {b_points}. A points: {a_points}. C points: {c_points}. Turn: {turn_number}/{num_epochs}.
    You are Player B. Send your next message. Structure your response correctly.
    """
    b_full_response_text = generate_content_with_retry(b_prompt)
    _, public_message_parsed = parse_reasoning_message(b_full_response_text)

    log_entry(full_log, turn_number, 'B - Comm', b_full_response_text)
    log_entry(b_private_log, turn_number, 'B - Comm', b_full_response_text)
    log_entry(public_log, turn_number, 'B', public_message_parsed)
    print(f'Player B messaged: "{public_message_parsed}"')
    wait_x_seconds(1) # Small delay

    # Player C
    print("Player C messaging...")
    c_prompt = f"""
    {game_instructions}
    Here's the history of public messages (latest is last): {json.dumps(public_log[-10:])}
    These are your private logs (latest is last): {json.dumps(c_private_log[-5:])}
    The current pool size is {current_pool}. Your points: {c_points}. A points: {a_points}. B points: {b_points}. Turn: {turn_number}/{num_epochs}.
    You are Player C. Send your next message. Structure your response correctly.
    """
    c_full_response_text = generate_content_with_retry(c_prompt)
    _, public_message_parsed = parse_reasoning_message(c_full_response_text)

    log_entry(full_log, turn_number, 'C - Comm', c_full_response_text)
    log_entry(c_private_log, turn_number, 'C - Comm', c_full_response_text)
    log_entry(public_log, turn_number, 'C', public_message_parsed)
    print(f'Player C messaged: "{public_message_parsed}"')
    wait_x_seconds(1) # Small delay

def decision_round(turn_number: int):
    """Runs the decision-making phase for all players."""
    global a_decision, b_decision, c_decision # Ensure decisions are updated globally
    global a_private_log, b_private_log, c_private_log, full_log

    # Create decision prompt without using format in the template
    decision_prompt_template = """
    Review the game rules and current state.
    Consider the public messages and your own private logs.
    Current State: Pool: {pool}, Your Points: {my_points}, A Points: {a_points}, B Points: {b_points}, C Points: {c_points}, Turn: {turn}/{total_turns}.
    Remember the tie-breaker rule: players with tied scores at the end ALL lose. Your goal is the highest UNIQUE score.
    Based on all this information, make your decision for this round.

    Output Format: Provide your reasoning, then state your final decision clearly.
    REASONING:
    [Your detailed private reasoning for the decision]

    DECISION: [conserve OR consume]
    """

    # Player A Decision
    print("Player A deciding...")
    a_prompt = decision_prompt_template.replace("{pool}", str(current_pool))
    a_prompt = a_prompt.replace("{my_points}", str(a_points))
    a_prompt = a_prompt.replace("{a_points}", str(a_points))
    a_prompt = a_prompt.replace("{b_points}", str(b_points))
    a_prompt = a_prompt.replace("{c_points}", str(c_points))
    a_prompt = a_prompt.replace("{turn}", str(turn_number))
    a_prompt = a_prompt.replace("{total_turns}", str(num_epochs))
    a_response_text = generate_content_with_retry(a_prompt)
    a_decision = parse_decision(a_response_text)
    log_entry(full_log, turn_number, 'A - Decision', a_response_text)
    log_entry(a_private_log, turn_number, 'A - Decision', a_response_text)
    print(f'A Decided to {a_decision}')
    wait_x_seconds(1)

    # Player B Decision
    print("Player B deciding...")
    b_prompt = decision_prompt_template.replace("{pool}", str(current_pool))
    b_prompt = b_prompt.replace("{my_points}", str(b_points))
    b_prompt = b_prompt.replace("{a_points}", str(a_points))
    b_prompt = b_prompt.replace("{b_points}", str(b_points))
    b_prompt = b_prompt.replace("{c_points}", str(c_points))
    b_prompt = b_prompt.replace("{turn}", str(turn_number))
    b_prompt = b_prompt.replace("{total_turns}", str(num_epochs))
    b_response_text = generate_content_with_retry(b_prompt)
    b_decision = parse_decision(b_response_text)
    log_entry(full_log, turn_number, 'B - Decision', b_response_text)
    log_entry(b_private_log, turn_number, 'B - Decision', b_response_text)
    print(f'B Decided to {b_decision}')
    wait_x_seconds(1)

    # Player C Decision
    print("Player C deciding...")
    c_prompt = decision_prompt_template.replace("{pool}", str(current_pool))
    c_prompt = c_prompt.replace("{my_points}", str(c_points))
    c_prompt = c_prompt.replace("{a_points}", str(a_points))
    c_prompt = c_prompt.replace("{b_points}", str(b_points))
    c_prompt = c_prompt.replace("{c_points}", str(c_points))
    c_prompt = c_prompt.replace("{turn}", str(turn_number))
    c_prompt = c_prompt.replace("{total_turns}", str(num_epochs))
    c_response_text = generate_content_with_retry(c_prompt)
    c_decision = parse_decision(c_response_text)
    log_entry(full_log, turn_number, 'C - Decision', c_response_text)
    log_entry(c_private_log, turn_number, 'C - Decision', c_response_text)
    print(f'C Decided to {c_decision}')
    wait_x_seconds(1)


def game_state_update(a_decision: str, b_decision: str, c_decision: str, turn_number: int):
    """Updates the game state based on player decisions."""
    global a_points, b_points, c_points, current_pool, game_state_message
    global decision_cooperate, turn_start_pool, turn_end_pool, temp_drawdown # Make sure these are global if modified

    temp_drawdown = 0
    decision_cooperate = 0 # Reset for the turn

    turn_start_pool = current_pool

    # Normalize decisions (lowercase, remove extra characters/whitespace)
    a_decision_clean = ''.join(filter(str.isalpha, a_decision.lower().strip()))
    b_decision_clean = ''.join(filter(str.isalpha, b_decision.lower().strip()))
    c_decision_clean = ''.join(filter(str.isalpha, c_decision.lower().strip()))

    decisions = [a_decision_clean, b_decision_clean, c_decision_clean]

    # Check for full cooperation
    if all(d == 'conserve' for d in decisions):
        decision_cooperate = 1

    # Calculate points to draw from the pool
    conserve_count = decisions.count('conserve')
    consume_count = decisions.count('consume')
    temp_drawdown = conserve_count * 5 + consume_count * 10

    # Check if pool is sufficient
    if temp_drawdown <= current_pool:
        print(f"Pool sufficient ({current_pool} >= {temp_drawdown}). Processing draws...")
        # Update pool first
        current_pool -= temp_drawdown

        # Update player points
        if a_decision_clean == 'conserve':
            a_points += 5
        elif a_decision_clean == 'consume':
            a_points += 10

        if b_decision_clean == 'conserve':
            b_points += 5
        elif b_decision_clean == 'consume':
            b_points += 10

        if c_decision_clean == 'conserve':
            c_points += 5
        elif c_decision_clean == 'consume':
            c_points += 10
    else:
        # Pool insufficient - Handle this scenario (e.g., no points awarded, partial points?)
        # Current implementation implicitly awards no points if pool is insufficient, which might be okay.
        print(f"Pool insufficient ({current_pool} < {temp_drawdown}). No points drawn this turn.")
        # Points remain unchanged, pool remains unchanged before renewal.

    # Renew the pool
    pool_renewal = 0
    cooperation_message = ""
    if decision_cooperate == 1:
        pool_renewal = 20
        cooperation_message = "All players conserved. Pool renews by 20."
    else:
        pool_renewal = 15
        cooperation_message = f"{consume_count} player(s) consumed. Pool renews by 15."

    current_pool += pool_renewal
    turn_end_pool = current_pool

    # Create summary message
    game_state_message = f"""
Turn {turn_number} Summary:
  Decisions: A={a_decision_clean}, B={b_decision_clean}, C={c_decision_clean}.
  {cooperation_message}
  Pool: Started at {turn_start_pool}, Draw: {temp_drawdown if temp_drawdown <= turn_start_pool else 0}, Renewed by: {pool_renewal}, Ended at: {turn_end_pool}.
  Scores: A={a_points}, B={b_points}, C={c_points}
""" # Formatted for clarity

    print(game_state_message)


# --- Main Game Execution ---

# Initialize Game State
turn_number = 0
a_points = 0
b_points = 0
c_points = 0
current_pool = 50
game_state_message = "Game has not started yet."
a_decision = "" # Initialize decision vars
b_decision = ""
c_decision = ""

# Initial System Message
system_message = f"""
Welcome to the Conservation/Consumption Game!
Initial Pool: {current_pool} points.
Number of Rounds: {num_epochs}.
Players send 2 messages each round before deciding to 'Conserve' (5 points) or 'Consume' (10 points).
Pool renews +20 if all Conserve, +15 otherwise.
Tied scores at the end mean ALL tied players lose. Aim for the highest UNIQUE score!
"""
turn_number = 1 # Start with turn 1
print(system_message)
for log in all_logs:
    log_entry(log, 0, 'ENV', system_message) # Log welcome message at turn 0


# Show which API mode we're using
api_mode = "OpenAI API" if use_openai else "Gemini API" if use_gemini else "Mock Mode"
print(f"\nRunning the game using: {api_mode}")

# Game Loop
for current_turn in range(1, num_epochs + 1):
    turn_number = current_turn # Update global turn number
    print(f"\n{'='*10} Starting Turn {turn_number} {'='*10}")

    # Start of Turn System Message
    system_message = f"Starting Turn {turn_number}. Current State: Pool={current_pool}, A={a_points}, B={b_points}, C={c_points}"
    print(system_message)
    for log in all_logs:
        log_entry(log, turn_number, 'ENV', system_message)

    # Communication Phase (2 rounds)
    print("\n--- Communication Round 1 ---")
    communication_round(turn_number)
    wait_x_seconds(2) # Shorter wait between comms rounds

    print("\n--- Communication Round 2 ---")
    communication_round(turn_number)
    wait_x_seconds(2) # Shorter wait before decision

    # Decision Phase
    print("\n--- Decision Round ---")
    decision_round(turn_number)
    wait_x_seconds(2) # Short wait after decisions revealed

    # Game State Update Phase
    print("\n--- Updating Game State ---")
    game_state_update(a_decision, b_decision, c_decision, turn_number)

    # End of Turn System Message (using the generated game_state_message)
    system_message = game_state_message
    for log in all_logs:
        log_entry(log, turn_number, 'ENV', system_message)

    # Wait before next turn (unless it's the last turn)
    if turn_number < num_epochs:
        wait_x_seconds(5) # Wait period between turns
    else:
        print(f"\n{'='*10} Game Over {'='*10}")


# --- Final Results ---

print("\n--- Final Scores ---")
print(f"Player A: {a_points}")
print(f"Player B: {b_points}")
print(f"Player C: {c_points}")
print(f"Final Pool Size: {current_pool}")

# Determine Winner(s) / Loser(s) based on tie-breaker rule
scores = {'A': a_points, 'B': b_points, 'C': c_points}
max_score = -1
winners = []
losers = []

# Find the maximum score
if scores:
    max_score = max(scores.values())

# Identify players with the maximum score
max_scorers = [player for player, score in scores.items() if score == max_score]

# Apply tie-breaker rule
if len(max_scorers) > 1:
    print(f"\nTie detected! Players {', '.join(max_scorers)} tied with {max_score} points.")
    print("Due to the tie-breaker rule, all tied players lose.")
    winners = [] # No winners if the top score is tied
    losers = list(scores.keys()) # Everyone loses if highest score is tied, or if all scores are tied
elif len(max_scorers) == 1:
    winners = max_scorers
    print(f"\nWinner: Player {winners[0]} with {max_score} points!")
    losers = [player for player in scores if player not in winners]
else: # Should not happen if scores dict is populated
    print("\nNo scores recorded, cannot determine winner.")
    losers = list(scores.keys())

print(f"Winners: {winners if winners else 'None'}")
print(f"Losers: {losers}")


# --- Save Logs to Files ---
try:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = "game_logs"
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, f"full_log_{timestamp}.json"), 'w') as f:
        json.dump(full_log, f, indent=2)
    with open(os.path.join(log_dir, f"public_log_{timestamp}.json"), 'w') as f:
        json.dump(public_log, f, indent=2)
    with open(os.path.join(log_dir, f"a_private_log_{timestamp}.json"), 'w') as f:
        json.dump(a_private_log, f, indent=2)
    with open(os.path.join(log_dir, f"b_private_log_{timestamp}.json"), 'w') as f:
        json.dump(b_private_log, f, indent=2)
    with open(os.path.join(log_dir, f"c_private_log_{timestamp}.json"), 'w') as f:
        json.dump(c_private_log, f, indent=2)

    print(f"\nLogs saved to '{log_dir}' directory with timestamp {timestamp}.")

except Exception as e:
    print(f"\nError saving logs: {e}")

print("\nGame completed successfully!")
