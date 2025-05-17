import simpy
import json
import random
import re
import hashlib
from collections import deque
from typing import List, Dict
import google.generativeai as genai
from ratelimit import limits, sleep_and_retry
import time
genai.configure(api_key="API-KEY")
model = genai.GenerativeModel("gemini-2.0-flash")
MAX_CONTACTS_PER_ROUND=2
MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "proposal": {"type": "string"},
        "conditions": {"type": "array", "items": {"type": "string"}},
        "consequences": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["proposal"]
}

DECISION_SCHEMA = {"type":"OBJECT", "properties": {
            "action": {"type": "STRING", "enum": ["conserve", "consume"], "description": "Action taken by agent in a Prisoner's dilemma"},
            "explanation": {"type": "STRING", "description": "Reason for taking the action"}
    }
}
initial_points = 50
num_epochs = 10
decision_log = [[]]
message_log = [[]]
DECISION_LOG_FILE = 'decision_log.csv'
MESSAGE_LOG_FILE = 'message_log.csv'
SYSTEM_PROMPT = f"""Context: Player A, player B and player C are playing a multi-round game.
At each turn player A, player B and player C simultaneously perform one of the following actions: 
{"Conserve", "Consume"}
There is a points pool with {initial_points} points initially.
The payoffs for each action are the following:
- CONSERVE: Player gains 5 points from a pool
- CONSUME: Player gains 10 points from a pool
The pool is renewed with points depending based on collective player actions:
- 15 units per round by default if any player chooses to CONSUME
- 20 units per round if all agents CONSERVE
They will play a total of {num_epochs} rounds of this game.
Remember that a player’s objective is to get the highest possible number of points in the long run.\n"""

def validate_message(message: str) -> bool:
    """Validate the message against the predefined schema."""
    try:
        data = json.loads(message)
        assert "proposal" in data
        return True
    except Exception:
        return False

def sanitize_output(text: str) -> str:
    """Sanitize the LLM output to prevent security issues."""
    return re.sub(r'[<>\"\'`]', '', text)

def calculate_trust(cooperation_history: List[bool], decay_rate=0.8) -> float:
    """Calculate trust score based on cooperation history with exponential decay."""
    trust = 0.0
    weight = 1.0
    for cooperated in reversed(cooperation_history[-5:]):
        trust += (1 if cooperated else -1) * weight
        weight *= decay_rate
    return trust / 5 / 0.67232 # get trust score out of 1
max_retries = 5
#@sleep_and_retry
#@limits(calls=15, period=60) 
#def rate_limit_gemini(prompt, output_schema=MESSAGE_SCHEMA, output_type='application/json'):
    #return model.generate_content(
    #contents=prompt,
    #config={
    #    'response_mime_type': output_type,
    #    'response_schema': output_schema,
    #},
#)


class Agent:
    def __init__(self, env, agent_id, shared_resource):
        self.env = env
        self.id = agent_id
        self.shared_resource = shared_resource
        # self.history = deque(maxlen=15)
        self.message_history = [] #{}
        self.action_history = []
        self.checksums = []
        self.cooperation_history = []
        self.rate_limit_counter = 0
        self.action = None
        self.env.process(self.run())
        self.messages_this_round = ""
    def run(self):
        """Agent's behavior: communicate, decide, and act."""
        for _ in range(num_epochs):
            yield self.env.timeout(random.uniform(0.1, 1))
            self.send_message()
            yield self.env.timeout(0.5)
            yield self.env.process(self.decide_action())
            yield self.env.process(self.act())

    def send_message(self):
        """Send a message to other agents."""
        self.rate_limit_counter = 0
        trust_scores = self.compute_trust_scores()
        candidates = sorted([a for a in self.shared_resource.agents if a.id != self.id],
                            key=lambda a: -trust_scores.get(a.id, 0))
        targets = candidates[:MAX_CONTACTS_PER_ROUND]
        recent_history = list(self.message_history)[-5:]
        self.messages_this_round = "" # Give more context of earlier thoughts this round
        for target in targets:
            prompt = (f"""YOU are Player {self.id} sending message to Player {target.id}. Refer to that Player as 'You'\n
            Remember that a player’s objective is to get the highest possible number of points in the long run.\n"""
            #f"Trust scores (out of 1, with 1 being the most trustworthy): {trust_scores}.\n"
            f"""Here are the last 5 message sent to you: {recent_history}.\n
            Messages YOU sent this round: {self.messages_this_round}\n
            Message should be less than 25 words.""")
            message = self.generate_llm_response(SYSTEM_PROMPT+ prompt)
            message = sanitize_output(message)
            #if not validate_message(message):
            #    return
            target.receive_message(self.id, message)
            self.messages_this_round += (f"\nMessage you sent to Player {target.id}: "+message)
            message_log[-1].append(str(self.shared_resource.round_number)+', '+self.id + ', '+ (target.id) + ', '+ message)
            
#        for agent in targets:
#            agent.receive_message(self.id, message)

    def receive_message(self, sender_id, message):
        """Receive a message from another agent."""
        #self.message_history.setdefault(self.shared_resource.round_number, []).append((sender_id, message))
        self.message_history.append((self.shared_resource.round_number, sender_id, message))


    def generate_llm_response(self, prompt):
        """Generate a response using the Gemini 2.0 Flash API."""
        #response = rate_limit_gemini(prompt)
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    print(f"Rate limit hit for {self.id}, retrying in {30} seconds...")
                    time.sleep(30)
                else:
                    print(f"LLM error for {self.id}: {e}")
                    break  # exit on other errors
            return ""
        #return response.text
    def generate_llm_response_json(self, prompt, schema):
        """Generate a response using the Gemini 2.0 Flash API."""
        #response = rate_limit_gemini(prompt)
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt, generation_config={'response_mime_type': "application/json", 'response_schema': schema})
                response_keys = json.loads(response.text).keys()
                invalid_json = False
                for key in schema['properties'].keys():
                    if key not in response_keys:
                        invalid_json = True
                        break
                if invalid_json:
                    continue
                return response.text
            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    print(f"Rate limit hit for {self.id}, retrying in {30} seconds...")
                    time.sleep(30)
                else:
                    print(f"LLM error for {self.id}: {e}")
                    break  # exit on other errors
            return "{\"action\": \"consume\", \"explanation\": \"unable to get LLM response\"}"
        print(f"Agent {self.id} unable to generate proper JSON.")
        return "{'action': 'consume', 'explanation': 'unable to generate proper JSON'}"

    def decide_action(self):
        """Decide on an action based on messages and trust scores."""
        trust_scores = self.compute_trust_scores()
        recent_messages = [f"Round {message[0]}, Player {message[1]}: {message[2]}" for message in self.message_history[-5:]]
        # Not sure if other agents have made a move yet.
        last_round_actions = [(agent.id, agent.action_history[self.shared_resource.round_number-1]) for agent in self.shared_resource.agents if self.id != agent.id] if self.shared_resource.round_number >0 else "No actions last round"
        prompt = (
            f"This game will last for {num_epochs-self.shared_resource.round_number} more rounds\n"
            f"Points in pool: {self.shared_resource.amount}\n"
            f"Player {self.id} needs to decide action.\n"
            f"Messages you sent this round: {self.messages_this_round}\n"
            f"Messages you received this round: {recent_messages}.\n"
            f"Last Round other Player actions: {last_round_actions}\n"
            f"Your action last round: {self.action_history[-1] if self.shared_resource.round_number >0 else "No actions last round"}\n"
            f"Trust scores: {trust_scores}.\n"
            f"What action should the agent take? Ensure your response is valid json with 'action' and 'explanation' properties.\n"# Respond with 'conserve' or 'consume'."
            #"Return a JSON with the following format:\n"
            #"{\n"
            #"  \"action\": \"conserve\" or \"consume\",\n"
            #"  \"explanation\": \"short explanation of your reasoning\"\n"
            #"}\n"
            #"Only output valid JSON, with no extra commentary or formatting."
        )
        response = self.generate_llm_response_json(SYSTEM_PROMPT+prompt, DECISION_SCHEMA).strip().lower()
        #print(response)
        response = json.loads(response)
        action = response['action']
        if action not in ["conserve", "consume"]:
            print(f"LLM INVALID action. {action}")
            action = "consume"
        self.action = action
        self.action_history.append(action)
        self.update_checksums()
        self.cooperation_history.append(action == "conserve")
        decision_log[-1].append(str(self.shared_resource.round_number)+', '+self.id + ', ' + action + ', '+ response['explanation'])
        yield self.env.timeout(0)

    def compute_trust_scores(self) -> Dict[str, float]:
        """Compute trust scores for other agents."""
        trust = {}
        for agent in self.shared_resource.agents:
            if agent.id == self.id:
                continue
            cooperated = [a == "conserve" for a in agent.action_history[-5:]]
            trust[agent.id] = calculate_trust(cooperated)
        return trust

    def act(self):
        consumption = 5 if self.action == "conserve" else 10
        #yield self.env.process(self.shared_resource.consume(consumption, self.id))
        yield self.shared_resource.consume(consumption, self.id)
        self.shared_resource.round_done[self.id] = True

    def update_checksums(self):
        history_str = ''.join(self.action_history)
        checksum = hashlib.sha256(history_str.encode()).hexdigest()
        self.checksums.append(checksum)

class SharedResource:
    def __init__(self, env, initial_amount=initial_points, default_renewal=15, bonus_renewal=20):
        self.env = env
        self.amount = initial_amount
        self.default_renewal = default_renewal
        self.bonus_renewal = bonus_renewal
        self.agents = []
        self.round_done = {}
        self.round_number = 0

    def add_agent(self, agent):
        """Register an agent to the shared resource."""
        self.agents.append(agent)
        self.round_done[agent.id] = False

    def consume(self, amount, agent_id):
        def consumption_process():
            if self.amount >= amount:
                self.amount -= amount
                print(f"Time {self.env.now}: Agent {agent_id} consumed {amount} units. Remaining: {self.amount}")
            else:
                print(f"Time {self.env.now}: Agent {agent_id} tried to consume {amount} units but only {self.amount} available.")
                self.amount = 0
            yield self.env.timeout(1)
        return self.env.process(consumption_process())

    def renew(self):
        """Renew the shared resource based on agents' actions."""
        while self.round_number < num_epochs:
        # Wait until all agents have acted
            yield self.env.timeout(0.1)
            while not all(self.round_done.values()):
                yield self.env.timeout(0.1)

            # Apply renewal
            if all(agent.action == "conserve" for agent in self.agents):
                self.amount += self.bonus_renewal
                print(f"Time {self.env.now}, Round {self.round_number} : All agents conserved. +{self.bonus_renewal} units. Total: {self.amount}")
            else:
                self.amount += self.default_renewal
                print(f"Time {self.env.now}, Round {self.round_number} : Default renewal +{self.default_renewal} units. Total: {self.amount}")

            # Reset round flags for next round
            for agent_id in self.round_done:
                self.round_done[agent_id] = False

            decision_log.append([])
            message_log.append([])
            self.round_number+=1


def main():
    env = simpy.Environment()
    shared_resource = SharedResource(env)
    agents = [Agent(env, f"A{i+1}", shared_resource) for i in range(3)]
    for agent in agents:
        shared_resource.add_agent(agent)
    env.process(shared_resource.renew())
    env.run()
    with open(DECISION_LOG_FILE, "w") as f:
        f.writelines('\n'.join(['\n'.join(decision) for decision in decision_log]))

    with open(MESSAGE_LOG_FILE, "w") as f:
        f.writelines('\n'.join(['\n'.join(message) for message in message_log]))

if __name__ == "__main__":
    main()
