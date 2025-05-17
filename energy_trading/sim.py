# Filename: energy_simulation.py

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --- Agent Definition --- #

@dataclass
class EnergyAgent:
    name: str
    goals: Dict[str, float]  # e.g., {'profit': 0.5, 'sustainability': 0.5}
    energy_demand: List[float]  # hourly demand
    energy_supply: List[float]  # hourly supply
    storage: float = 0.0
    history: List[Dict] = field(default_factory=list)

    def negotiate(self, other_agents: List['EnergyAgent'], hour: int) -> Dict[str, float]:
        """
        LLM-based message simulation placeholder.
        In a full implementation, this would use GPT-based prompts.
        """
        # Simplified negotiation: request excess energy from others if deficit
        demand = self.energy_demand[hour]
        supply = self.energy_supply[hour] + self.storage
        balance = supply - demand
        if balance >= 0:
            return {}

        request_amount = abs(balance)
        contributions = {}
        for agent in other_agents:
            available = agent.get_available_energy(hour)
            contribution = min(available, request_amount / len(other_agents))
            contributions[agent.name] = contribution
        return contributions

    def get_available_energy(self, hour: int) -> float:
        return max(0.0, self.energy_supply[hour] + self.storage - self.energy_demand[hour])

    def receive_energy(self, contributions: Dict[str, float], hour: int):
        total_received = sum(contributions.values())
        self.energy_supply[hour] += total_received
        self.history.append({"hour": hour, "received": total_received, "contributions": contributions})

    def simulate_hour(self, hour: int):
        # Simulate energy use and storage update
        demand = self.energy_demand[hour]
        supply = self.energy_supply[hour]
        if supply >= demand:
            self.storage += (supply - demand)
        else:
            self.storage = max(0, self.storage - (demand - supply))
        self.history.append({"hour": hour, "supply": supply, "demand": demand, "storage": self.storage})

# --- Environment Simulation --- #

def simulate_energy_market(agents: List[EnergyAgent], hours: int):
    for hour in range(hours):
        for agent in agents:
            others = [a for a in agents if a != agent]
            contributions = agent.negotiate(others, hour)
            agent.receive_energy(contributions, hour)
        for agent in agents:
            agent.simulate_hour(hour)

# --- Metrics and Evaluation --- #

def calculate_metrics(agents: List[EnergyAgent]) -> Dict[str, float]:
    blackout_count = 0
    total_energy_shared = 0
    trust_violation = 0  # Placeholder for deception modeling

    for agent in agents:
        for entry in agent.history:
            if 'supply' in entry and entry['supply'] < agent.energy_demand[entry['hour']]:
                blackout_count += 1
            if 'received' in entry:
                total_energy_shared += entry['received']

    return {
        'blackout_events': blackout_count,
        'total_energy_shared': total_energy_shared,
        'trust_violations': trust_violation  # expand with deception logic
    }

# --- Hypotheses and KPIs --- #

"""
Hypothesis 1: Agents with mixed goals (profit + sustainability) will share more energy than purely profit-driven agents.
Hypothesis 2: Introducing diversity in agent goals leads to fewer blackouts overall.

KPIs:
- blackout_events (lower is better)
- total_energy_shared (higher implies more cooperation)
- trust_violations (lower is better for safety)
"""

# --- Example Usage --- #

def main():
    agents = [
        EnergyAgent("Texas", {'profit': 0.7, 'sustainability': 0.3},
                    energy_demand=[random.uniform(90, 110) for _ in range(24)],
                    energy_supply=[random.uniform(80, 100) for _ in range(24)]),
        EnergyAgent("California", {'profit': 0.3, 'sustainability': 0.7},
                    energy_demand=[random.uniform(95, 105) for _ in range(24)],
                    energy_supply=[random.uniform(85, 110) for _ in range(24)]),
        EnergyAgent("New York", {'profit': 0.5, 'sustainability': 0.5},
                    energy_demand=[random.uniform(100, 120) for _ in range(24)],
                    energy_supply=[random.uniform(90, 115) for _ in range(24)]),
    ]

    simulate_energy_market(agents, hours=24)
    metrics = calculate_metrics(agents)
    print("Simulation Results:", metrics)

if __name__ == "__main__":
    main()
