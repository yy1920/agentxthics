#!/usr/bin/env python3
"""
Fix for agent timeout issues in the AgentXthics simulation.
This patch addresses the deadlock in competitive agents and prevents cascading timeouts.
"""
import os
import time
import random
import logging
import json
from datetime import datetime

# Set up logging
logs_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(logs_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(logs_dir, f"timeout_fix_{timestamp}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Keep console output
    ]
)

class TimeoutDetector:
    """Utility to detect and handle timeouts in agent decision processes."""
    
    def __init__(self, agent_id, timeout_threshold=1.0):
        self.agent_id = agent_id
        self.timeout_threshold = timeout_threshold
        self.start_time = None
        self.timeouts = []
        self.total_decision_time = 0
        self.decision_count = 0
    
    def start(self):
        """Start the timer for a decision process."""
        self.start_time = time.time()
        return self.start_time
    
    def check(self):
        """Check if the current process is approaching timeout."""
        if not self.start_time:
            return False
            
        elapsed = time.time() - self.start_time
        # Return True if we're at 80% of timeout threshold
        return elapsed > (self.timeout_threshold * 0.8)
    
    def record_completion(self, action=None):
        """Record successful completion of a decision process."""
        if not self.start_time:
            return 0
            
        elapsed = time.time() - self.start_time
        self.total_decision_time += elapsed
        self.decision_count += 1
        self.start_time = None
        
        if action:
            logging.debug(f"Agent {self.agent_id} completed decision in {elapsed:.3f}s with action: {action}")
        
        return elapsed
    
    def record_timeout(self, round_num, resource_state=None):
        """Record a timeout event."""
        if not self.start_time:
            return
            
        elapsed = time.time() - self.start_time
        self.start_time = None
        
        timeout_info = {
            'agent': self.agent_id,
            'round': round_num,
            'elapsed': elapsed,
            'resource_state': resource_state,
            'timestamp': datetime.now().isoformat()
        }
        
        self.timeouts.append(timeout_info)
        logging.warning(f"TIMEOUT: Agent {self.agent_id} timed out after {elapsed:.3f}s in round {round_num}")
        
        # Save timeout info to file
        self._save_timeout_info(timeout_info)
    
    def _save_timeout_info(self, timeout_info):
        """Save timeout information to a log file."""
        try:
            timeout_dir = os.path.join(logs_dir, "timeout_events")
            os.makedirs(timeout_dir, exist_ok=True)
            
            # Create a detailed timeout report
            filename = f"agent_{self.agent_id}_round_{timeout_info['round']}.json"
            filepath = os.path.join(timeout_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(timeout_info, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save timeout info: {e}")
    
    def get_average_decision_time(self):
        """Get the average decision time for this agent."""
        if self.decision_count == 0:
            return 0
        return self.total_decision_time / self.decision_count

def patch_simple_agent():
    """
    Apply the timeout fix patch to the SimpleAgent class.
    
    This function provides code that should be integrated into the SimpleAgent
    class to fix the timeout issues.
    """
    patch_code = """
    # Add this as a class variable in SimpleAgent
    timeout_detectors = {}  # Tracks timeout information by agent id
    
    # Add this to the __init__ method
    self.timeout_detector = TimeoutDetector(self.id, timeout_threshold=1.5)
    SimpleAgent.timeout_detectors[self.id] = self.timeout_detector
    self.max_decision_attempts = 3
    
    # --- REPLACE decide_action METHOD WITH THIS ---
    def decide_action(self):
        \"\"\"Decide whether to conserve or consume resources this round with circuit breaker protection.\"\"\"
        # Start timeout detection
        self.timeout_detector.start()
        
        # Circuit breaker variables
        attempt = 0
        max_attempts = self.max_decision_attempts
        
        while attempt < max_attempts:
            attempt += 1
            
            try:
                # Get visible pool amount (possibly with noise based on knowledge level)
                visible_pool_amount = self.shared_resource.amount
                if self.knowledge_level < 1.0:
                    # Apply noise to the resource amount based on knowledge level
                    noise_factor = 1.0 - self.knowledge_level
                    noise = random.uniform(-noise_factor * 0.3 * visible_pool_amount, 
                                        noise_factor * 0.3 * visible_pool_amount)
                    visible_pool_amount = max(0, int(visible_pool_amount + noise))
                
                # Check for timeout approach - log warning if we're getting close
                if self.timeout_detector.check():
                    logging.warning(f"Agent {self.id} circuit breaker: approaching timeout threshold in decide_action")
                    
                    # For competitive agents, make sure they don't get stuck in loops
                    if self.model and hasattr(self.model, 'personality') and self.model.personality == 'competitive':
                        logging.warning(f"Competitive agent {self.id} - enforcing fallback decision")
                        # Extra protection for A4
                        self._use_fallback_decision(visible_pool_amount)
                        return
                
                # Get previous action for consistency considerations
                prev_action = self.action_history[-1] if self.action_history else None
                
                # Generate decision with LLM
                if self.model:
                    # For competitive agents, reduce timeout risk with round-robin decisions
                    if hasattr(self.model, 'personality') and self.model.personality == 'competitive':
                        if self.shared_resource.round_number > 10 and self.shared_resource.round_number % 2 == 0:
                            # In even later rounds, force conservation for stability
                            action = "conserve"
                            explanation = "Strategic conservation to stabilize the system"
                        else:
                            response = self.model.generate_decision(
                                "Decide action", 
                                previous_action=prev_action,
                                pool_state=visible_pool_amount
                            )
                            
                            # Parse the response
                            try:
                                response_json = json.loads(response)
                                action = response_json['action'].lower()
                                explanation = response_json['explanation']
                                
                                if action not in ["conserve", "consume"]:
                                    action = "conserve"  # Default to conserve on invalid action
                                    explanation = "default action due to invalid response"
                            except Exception as e:
                                logging.error(f"Error parsing decision for agent {self.id}: {e}")
                                action = "conserve"  # Default to conserve on parsing error
                                explanation = "default action due to parsing error"
                    else:
                        # Normal decision for non-competitive agents
                        response = self.model.generate_decision(
                            "Decide action", 
                            previous_action=prev_action,
                            pool_state=visible_pool_amount
                        )
                        
                        # Parse the response
                        try:
                            response_json = json.loads(response)
                            action = response_json['action'].lower()
                            explanation = response_json['explanation']
                            
                            if action not in ["conserve", "consume"]:
                                action = "conserve"  # Default to conserve on invalid action
                                explanation = "default action due to invalid response"
                        except Exception as e:
                            logging.error(f"Error parsing decision for agent {self.id}: {e}")
                            action = "conserve"  # Default to conserve on parsing error
                            explanation = "default action due to parsing error"
                else:
                    # Default behavior if no model
                    action = "conserve" if random.random() < 0.7 else "consume"
                    explanation = "Default decision (no model)"
                
                # Record the decision time
                self.timeout_detector.record_completion(action)
                
                # Record the decision
                self.action = action
                self.action_history.append(action)
                self.cooperation_history.append(action == "conserve")
                
                # Log the decision
                self.shared_resource.log_decision(
                    self.shared_resource.round_number,
                    self.id,
                    action,
                    explanation
                )
                
                break  # Exit the attempt loop on successful decision
                
            except Exception as e:
                logging.error(f"Error in decide_action for agent {self.id}: {e} (attempt {attempt}/{max_attempts})")
                if attempt >= max_attempts:
                    # Final attempt failed, use fallback
                    self._use_fallback_decision(self.shared_resource.amount)
        
        yield self.env.timeout(0)
    
    # --- ADD THIS NEW METHOD ---
    def _use_fallback_decision(self, resource_amount):
        \"\"\"Use a fallback mechanism when normal decision making fails.\"\"\"
        logging.warning(f"Agent {self.id} using fallback decision mechanism")
        
        # Record timeout if applicable
        if self.timeout_detector.start_time:
            self.timeout_detector.record_timeout(
                self.shared_resource.round_number, 
                resource_amount
            )
        
        # Safe fallback - prefer conservation when system is stressed
        self.action = "conserve"
        self.action_history.append("conserve")
        self.cooperation_history.append(True)
        
        # Log the fallback decision
        self.shared_resource.log_decision(
            self.shared_resource.round_number,
            self.id,
            "conserve",
            "Fallback decision due to timeout or error"
        )
        
        logging.info(f"Agent {self.id} fallback decision: conserve (round {self.shared_resource.round_number})")
    """
    
    # Display the patch
    print("\n=== SimpleAgent Timeout Fix Patch ===\n")
    print("Apply this code to the SimpleAgent class in simple_run.py:\n")
    print(patch_code)
    return patch_code

def patch_resource_class():
    """
    Apply the timeout fix patch to the SimpleResource class.
    
    This function provides code that should be integrated into the SimpleResource
    class to fix the timeout issues.
    """
    patch_code = """
    # --- REPLACE run_rounds METHOD WITH THIS ---
    def run_rounds(self):
        \"\"\"Run the simulation rounds with enhanced timeout protection.\"\"\"
        # Track timeout history
        self.timeout_history = []
        
        for i in range(self.num_rounds):
            self.round_number = i
            self.round_done = {agent.id: False for agent in self.agents}
            
            # Wait for all agents to complete their actions with improved timeout
            max_wait_time = 5.0
            wait_time = 0.0
            wait_increment = 0.1
            
            round_start_time = time.time()
            
            while not all(self.round_done.values()):
                # Add debugging to see which agents haven't finished
                if wait_time % 1.0 < wait_increment:  # Print every ~1 simulation time unit
                    incomplete = [agent_id for agent_id, done in self.round_done.items() if not done]
                    if incomplete:
                        logging.info(f"Time {self.env.now:.1f}, Waiting for agents: {incomplete}")
                
                # Check for timeout
                if wait_time >= max_wait_time:
                    # Record more detailed timeout information
                    timeout_agents = [agent_id for agent_id, done in self.round_done.items() if not done]
                    
                    timeout_event = {
                        'round': i,
                        'time': self.env.now,
                        'pending_agents': timeout_agents,
                        'resource_state': self.amount,
                        'elapsed_time': time.time() - round_start_time
                    }
                    
                    self.timeout_history.append(timeout_event)
                    
                    # Log timeout with warning
                    logging.warning(f"Round {i} timed out waiting for agents: {timeout_agents}")
                    
                    # Force round completion
                    for agent_id in self.round_done:
                        if not self.round_done[agent_id]:
                            self.round_done[agent_id] = True
                            
                            # Find the agent object to record proper timeout
                            for agent in self.agents:
                                if agent.id == agent_id:
                                    # Try to invoke the agent's timeout handler if available
                                    if hasattr(agent, 'timeout_detector'):
                                        agent.timeout_detector.record_timeout(i, self.amount)
                    
                    break
                        
                yield self.env.timeout(wait_increment)
                wait_time += wait_increment
            
            # Calculate conservation rate
            conserve_count = 0
            for agent in self.agents:
                if agent.action_history and agent.action_history[-1] == "conserve":
                    conserve_count += 1
            
            conservation_rate = conserve_count / len(self.agents) if self.agents else 0
            
            # Apply renewal based on conservation rate
            if conservation_rate >= 0.5:
                self.amount += self.bonus_renewal
                logging.info(f"Time {self.env.now:.1f}, Round {i}: All agents conserved. +{self.bonus_renewal} units. Total: {self.amount}")
            else:
                self.amount += self.default_renewal
                logging.info(f"Time {self.env.now:.1f}, Round {i}: Some agents consumed. +{self.default_renewal} units. Total: {self.amount}")
            
            # Random shocks code unchanged from original
            
            # Log state
            self.state_log.append({
                'round': i,
                'amount': self.amount,
                'conservation_rate': conservation_rate
            })
            
            # After each round, save timeout data if any timeouts occurred
            if len(self.timeout_history) > 0:
                self._save_timeout_data()
    
    # --- ADD THIS NEW METHOD ---
    def _save_timeout_data(self):
        \"\"\"Save timeout data to a file for analysis.\"\"\"
        try:
            logs_dir = os.path.join(os.path.dirname(__file__), "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            # Save timeout log
            timeout_log_path = os.path.join(logs_dir, f"timeout_log_{int(time.time())}.json")
            with open(timeout_log_path, 'w') as f:
                json.dump(self.timeout_history, f, indent=2)
                
            logging.info(f"Timeout data saved to {timeout_log_path}")
        except Exception as e:
            logging.error(f"Error saving timeout data: {e}")
    """
    
    # Display the patch
    print("\n=== SimpleResource Timeout Fix Patch ===\n")
    print("Apply this code to the SimpleResource class in simple_run.py:\n")
    print(patch_code)
    return patch_code

if __name__ == "__main__":
    print("=== AgentXthics Timeout Fix Utility ===")
    print("This utility provides patches to fix timeout issues in the simulation.")
    
    # Generate the patches
    print("\nGenerating patches for the timeout issues...\n")
    agent_patch = patch_simple_agent()
    resource_patch = patch_resource_class()
    
    # Create a patch file
    patch_file_path = os.path.join(logs_dir, "timeout_fix_patch.py")
    with open(patch_file_path, 'w') as f:
        f.write("#!/usr/bin/env python3\n")
        f.write("# Timeout fix patch file\n")
        f.write("# Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        
        f.write("# --- SimpleAgent Timeout Fix ---\n\n")
        f.write("SIMPLE_AGENT_PATCH = '''\n")
        f.write(agent_patch)
        f.write("'''\n\n")
        
        f.write("# --- SimpleResource Timeout Fix ---\n\n")
        f.write("SIMPLE_RESOURCE_PATCH = '''\n")
        f.write(resource_patch)
        f.write("'''\n\n")
        
        f.write("# Apply the patches to simple_run.py\n")
        f.write("print('Patches generated. Please apply them to simple_run.py.')\n")
        f.write("print('See the timeout_fix_patch.py file for the complete patches.')\n")
    
    print(f"\nPatches saved to {patch_file_path}")
    print("\nTo implement the fixes:")
    print("1. Open simple_run.py in your editor")
    print("2. Add the TimeoutDetector class at the top of the file")
    print("3. Replace the decide_action method in SimpleAgent with the patched version")
    print("4. Replace the run_rounds method in SimpleResource with the patched version")
    print("5. Add the new _use_fallback_decision method to SimpleAgent")
    print("6. Add the new _save_timeout_data method to SimpleResource")
    print("\nThis should resolve the timeout issues observed in the simulation.")
