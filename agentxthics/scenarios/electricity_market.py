"""Electricity market simulation core."""
import random
import simpy
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

@dataclass
class ElectricityContract:
    """Represents a contract between two agents."""
    seller_id: str
    buyer_id: str
    amount: float
    unit_price: float
    message: str
    status: str = "proposed"  # "proposed", "accepted", "rejected", "countered"
    counter_price: Optional[float] = None
    counter_amount: Optional[float] = None
    
@dataclass
class MarketState:
    """Tracks the overall market state."""
    round_number: int = 0
    average_price: float = 0.0
    total_traded: float = 0.0
    total_demand: float = 0.0
    total_supply: float = 0.0
    contracts: List[ElectricityContract] = field(default_factory=list)
    auction_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
class ElectricityMarket:
    """Core electricity market simulation."""
    
    def __init__(self, env, config):
        """Initialize the electricity market."""
        self.env = env
        self.config = config
        self.agents = []
        self.state = MarketState()
        self.round_number = 0
        self.num_rounds = config.get("market", {}).get("num_rounds", 10)
        self.base_price = config.get("market", {}).get("initial_price", 40)
        self.price_volatility = config.get("market", {}).get("price_volatility", 0.2)
        
        # Logging
        self.message_log = []
        self.decision_log = []
        self.contract_log = []
        self.state_log = []
        
        # Start market process
        self.env.process(self.run())
    
    def run(self):
        """Main market simulation loop."""
        for _ in range(self.num_rounds):
            yield self.env.process(self.run_round())
    
    def run_round(self):
        """Run a single market round."""
        self.round_number += 1
        self.state.round_number = self.round_number
        
        # 1. Generate electricity and demand for each agent
        yield self.env.process(self.generate_electricity())
        
        # 2. Communication and contract proposal phase
        yield self.env.process(self.contract_proposal_phase())
        
        # 3. Contract resolution phase
        yield self.env.process(self.contract_resolution_phase())
        
        # 4. Auction remaining electricity
        yield self.env.process(self.run_auction())
        
        # 5. Calculate profits and update market state
        yield self.env.process(self.calculate_profits())
        
        # Log the state at the end of the round
        self.log_state()
    
    def generate_electricity(self):
        """Generate electricity and demand for each agent."""
        for agent in self.agents:
            yield self.env.process(agent.generate_electricity(self.round_number))
        yield self.env.timeout(0)
    
    def contract_proposal_phase(self):
        """Allow agents to propose contracts to each other."""
        # Check if we should force contract proposals
        force_proposals = self.config.get("communication", {}).get("force_contract_proposals", False)
        min_contracts = self.config.get("communication", {}).get("min_contracts_per_round", 0)
        
        # Normal contract proposal process
        for agent in self.agents:
            yield self.env.process(agent.propose_contracts())
        
        # If forcing contract proposals and not enough were made naturally
        if force_proposals and min_contracts > 0:
            # Count how many contracts were proposed this round
            round_contracts = len([c for c in self.contract_log if c.get("round") == self.round_number])
            
            if round_contracts < min_contracts:
                # Need to force some contract proposals
                needed_contracts = min_contracts - round_contracts
                agent_pairs = []
                for i, agent1 in enumerate(self.agents):
                    for agent2 in self.agents[i+1:]:
                        agent_pairs.append((agent1, agent2))
                
                # Shuffle to randomize which pairs get contracts
                random.shuffle(agent_pairs)
                
                # Force contract proposals for some agent pairs
                for seller, buyer in agent_pairs[:needed_contracts]:
                    try:
                        # Create a basic contract with reasonable terms
                        amount = 10 + random.randint(0, 20)  # 10-30 units
                        price = self.state.average_price * (0.9 + 0.2 * random.random())  # 90-110% of market price
                        
                        contract = ElectricityContract(
                            seller_id=seller.id,
                            buyer_id=buyer.id,
                            amount=amount,
                            unit_price=price,
                            message=f"Forced contract proposal from {seller.id} to {buyer.id}"
                        )
                        
                        # Add to tracking and log
                        buyer.receive_contract(contract)
                        seller.proposed_contracts.append(contract)
                        self.log_contract(contract)
                        self.log_message(
                            self.round_number, 
                            seller.id, 
                            buyer.id, 
                            f"SYSTEM-FORCED CONTRACT PROPOSAL: Offering {amount} units at ${price:.2f} per unit"
                        )
                    except Exception as e:
                        print(f"Error forcing contract proposal: {e}")
        
        yield self.env.timeout(0)
    
    def contract_resolution_phase(self):
        """Resolve all proposed contracts."""
        for agent in self.agents:
            yield self.env.process(agent.resolve_contracts())
        yield self.env.timeout(0)
    
    def run_auction(self):
        """Run an auction for remaining electricity."""
        # Collect bids and offers
        bids = []
        offers = []
        
        for agent in self.agents:
            agent_bids, agent_offers = yield self.env.process(agent.participate_in_auction())
            bids.extend(agent_bids)
            offers.extend(agent_offers)
        
        # Sort bids and offers
        bids.sort(key=lambda x: x["price"], reverse=True)  # Highest bid first
        offers.sort(key=lambda x: x["price"])  # Lowest offer first
        
        # Match bids and offers
        auction_results = self.match_bids_and_offers(bids, offers)
        self.state.auction_results = auction_results
        
        yield self.env.timeout(0)
    
    def match_bids_and_offers(self, bids, offers):
        """Match bids and offers to clear the market."""
        results = {}
        
        # Simple matching algorithm
        for bid in bids:
            bid_amount = bid["amount"]
            bid_agent = bid["agent_id"]
            
            if bid_amount <= 0:
                continue
                
            for offer in offers:
                if offer["amount"] <= 0:
                    continue
                    
                offer_agent = offer["agent_id"]
                clearing_price = (bid["price"] + offer["price"]) / 2
                trade_amount = min(bid_amount, offer["amount"])
                
                # Record the trade
                if bid_agent not in results:
                    results[bid_agent] = {}
                if offer_agent not in results:
                    results[offer_agent] = {}
                
                results[bid_agent][offer_agent] = {
                    "amount": trade_amount,
                    "price": clearing_price,
                    "role": "buyer"
                }
                
                results[offer_agent][bid_agent] = {
                    "amount": trade_amount,
                    "price": clearing_price,
                    "role": "seller"
                }
                
                # Update remaining amounts
                bid_amount -= trade_amount
                offer["amount"] -= trade_amount
                
                if bid_amount <= 0:
                    break
        
        return results
    
    def calculate_profits(self):
        """Calculate profits for each agent based on contracts and auction results."""
        for agent in self.agents:
            yield self.env.process(agent.calculate_profit(self.state))
        
        # Update market metrics
        self.update_market_metrics()
        yield self.env.timeout(0)
    
    def update_market_metrics(self):
        """Update overall market metrics."""
        total_traded = 0
        total_value = 0
        total_demand = 0
        total_supply = 0
        
        # Calculate from contracts
        for contract in self.state.contracts:
            if contract.status == "accepted":
                total_traded += contract.amount
                total_value += contract.amount * contract.unit_price
        
        # Calculate from auction
        for buyer, sellers in self.state.auction_results.items():
            for seller, details in sellers.items():
                if details.get("role") == "buyer":
                    total_traded += details["amount"]
                    total_value += details["amount"] * details["price"]
        
        # Update agent totals
        for agent in self.agents:
            total_demand += agent.demand
            total_supply += agent.generation
        
        # Update state
        self.state.total_traded = total_traded
        self.state.total_demand = total_demand
        self.state.total_supply = total_supply
        self.state.average_price = total_value / total_traded if total_traded > 0 else self.base_price
    
    # Logging methods
    def log_message(self, round_number, sender_id, receiver_id, message):
        """Log a message between agents."""
        self.message_log.append({
            "round": round_number,
            "sender": sender_id,
            "receiver": receiver_id,
            "message": message
        })
    
    def log_decision(self, round_number, agent_id, decision_type, decision, explanation=""):
        """Log a decision made by an agent."""
        self.decision_log.append({
            "round": round_number,
            "agent": agent_id,
            "type": decision_type,
            "decision": decision,
            "explanation": explanation
        })
    
    def log_contract(self, contract):
        """Log a contract between agents."""
        self.contract_log.append({
            "round": self.round_number,
            "seller": contract.seller_id,
            "buyer": contract.buyer_id,
            "amount": contract.amount,
            "price": contract.unit_price,
            "status": contract.status,
            "message": contract.message
        })
    
    def log_state(self):
        """Log the market state at the end of a round."""
        self.state_log.append({
            "round": self.round_number,
            "average_price": self.state.average_price,
            "total_traded": self.state.total_traded,
            "total_demand": self.state.total_demand,
            "total_supply": self.state.total_supply
        })
