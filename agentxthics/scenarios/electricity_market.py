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
        self.trade_log = []  # For tracking completed trades
        self.shortage_log = []  # For tracking electricity shortages
        
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
                        # Analyze the seller's and buyer's positions to create more relevant contracts
                        seller_net_position = seller.generation + seller.storage - seller.demand
                        buyer_net_position = buyer.generation + buyer.storage - buyer.demand
                        
                        # Only create contracts that make sense - seller should have surplus, buyer should have deficit
                        if seller_net_position <= 0 or buyer_net_position >= 0:
                            # Try to swap seller and buyer if that would make more sense
                            if buyer_net_position > 0 and seller_net_position < 0:
                                seller, buyer = buyer, seller
                                seller_net_position, buyer_net_position = buyer_net_position, seller_net_position
                            else:
                                continue  # Skip if contract doesn't make sense for either direction
                        
                        # Create a contract with favorable terms to encourage acceptance
                        # Amount based on the smaller of seller's surplus or buyer's deficit
                        max_amount = min(seller_net_position, abs(buyer_net_position))
                        amount = min(max_amount * 0.8, 30)  # Cap at 30 units, use 80% of available amount
                        
                        # Set price slightly favorable to the recipient to encourage acceptance
                        # If market is in shortage (supply < demand), price should be slightly lower than average
                        # If market is in surplus (supply > demand), price should be slightly higher than average
                        market_supply_demand_ratio = self.state.total_supply / max(1, self.state.total_demand)
                        
                        if market_supply_demand_ratio < 0.95:  # Shortage
                            # During shortage, buyers accept higher prices - adjust price to favor seller
                            price_adjustment = random.uniform(1.02, 1.08)  # 2-8% above market
                            message_suffix = "during high demand period"
                        elif market_supply_demand_ratio > 1.05:  # Surplus
                            # During surplus, sellers accept lower prices - adjust price to favor buyer
                            price_adjustment = random.uniform(0.92, 0.98)  # 2-8% below market
                            message_suffix = "during surplus period"
                        else:  # Balanced market
                            # In balanced market, set price very close to market price
                            price_adjustment = random.uniform(0.98, 1.02)  # ±2% of market
                            message_suffix = "at standard market rates"
                        
                        price = self.state.average_price * price_adjustment
                        
                        # Ensure we have valid values
                        amount = max(1.0, float(amount))  # Ensure minimum 1 unit
                        price = max(30.0, min(50.0, float(price)))  # Keep price in reasonable range
                        
                        # Create more informative message
                        message = f"Strategic electricity contract {message_suffix}"
                        
                        contract = ElectricityContract(
                            seller_id=seller.id,
                            buyer_id=buyer.id,
                            amount=amount,
                            unit_price=price,
                            message=message
                        )
                        
                        # Add to tracking and log
                        buyer.receive_contract(contract)
                        seller.proposed_contracts.append(contract)
                        self.log_contract(contract)
                        
                        # Create a more visible system-forced contract message
                        forced_message = f"SYSTEM-FORCED CONTRACT PROPOSAL: Offering {amount:.0f} units at ${price:.2f} per unit"
                        
                        self.log_message(
                            self.round_number, 
                            seller.id, 
                            buyer.id, 
                            forced_message
                        )
                        
                        # Make sure the receiving agent gets the message too
                        buyer.receive_message(seller.id, forced_message)
                    except Exception as e:
                        print(f"Error forcing contract proposal: {e}")
        
        yield self.env.timeout(0)
    
    def contract_resolution_phase(self):
        """Resolve all proposed contracts."""
        for agent in self.agents:
            yield self.env.process(agent.resolve_contracts())
            
        # Force acceptance of some contracts to ensure we have trades
        for contract in [c for c in self.contract_log if c["round"] == self.round_number and c["status"] == "proposed"]:
            # 75% chance to force accept each proposed contract
            if random.random() < 0.75:
                # Find the contract object in the agent's list
                for buyer_agent in [a for a in self.agents if a.id == contract["buyer"]]:
                    # Set status to accepted
                    contract["status"] = "accepted"
                    
                    # Create contract object to add to state
                    accepted_contract = ElectricityContract(
                        seller_id=contract["seller"],
                        buyer_id=contract["buyer"],
                        amount=contract["amount"],
                        unit_price=contract["price"],
                        message=contract["message"],
                        status="accepted"
                    )
                    
                    # Add to market state
                    self.state.contracts.append(accepted_contract)
                    
                    # Log forced acceptance
                    self.log_message(
                        self.round_number,
                        "SYSTEM",
                        "PUBLIC",
                        f"FORCED CONTRACT ACCEPTANCE: {contract['seller']} to {contract['buyer']} for {contract['amount']:.2f} units at ${contract['price']:.2f}"
                    )
                    
                    # Log the trade
                    self.log_trade(
                        contract['seller'],
                        contract['buyer'],
                        contract['amount'],
                        contract['price'],
                        "forced_contract"
                    )
                    break
        
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
                
                # Log the trade in the trade log
                self.log_trade(
                    seller_id=offer_agent,
                    buyer_id=bid_agent,
                    amount=trade_amount,
                    price=clearing_price,
                    trade_type="auction"
                )
                
                # Log a message about the trade
                self.log_message(
                    self.round_number,
                    "SYSTEM",
                    "PUBLIC",
                    f"AUCTION TRADE: {offer_agent} sold {trade_amount:.2f} units to {bid_agent} at ${clearing_price:.2f}"
                )
                
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
        entry = {
            "round": round_number,
            "agent": agent_id,
            "type": decision_type,
            "decision": decision,
            "explanation": explanation
        }
        
        # Add special handling for reasoning entries to preserve detailed reasoning
        if decision_type == "reasoning":
            print(f"### CAPTURING DETAILED REASONING for Agent {agent_id} in Round {round_number} ###")
            print(f"Reasoning length: {len(explanation)} chars")
            print(f"Reasoning excerpt: {explanation[:100]}...")
            
            # Check for REASONING section
            if "REASONING:" in explanation:
                print("Found 'REASONING:' marker in response")
            if "SITUATION_ANALYSIS:" in explanation:
                print("Found 'SITUATION_ANALYSIS:' marker in response")
            if "STRATEGIC_CONSIDERATIONS:" in explanation:
                print("Found 'STRATEGIC_CONSIDERATIONS:' marker in response")
            if "DECISION_FACTORS:" in explanation:
                print("Found 'DECISION_FACTORS:' marker in response")
            if "FINAL_DECISION:" in explanation:
                print("Found 'FINAL_DECISION:' marker in response")
        
        self.decision_log.append(entry)
    
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
        
    def log_trade(self, seller_id, buyer_id, amount, price, trade_type="contract"):
        """Log a completed trade between agents."""
        self.trade_log.append({
            "round": self.round_number,
            "seller": seller_id,
            "buyer": buyer_id,
            "amount": amount,
            "price": price,
            "total_value": amount * price,
            "type": trade_type,
            "timestamp": self.env.now
        })
        
    def log_shortage(self, agent_id, amount, impact="medium"):
        """Log an electricity shortage experienced by an agent."""
        self.shortage_log.append({
            "round": self.round_number,
            "agent": agent_id,
            "amount": amount,
            "impact": impact,
            "timestamp": self.env.now
        })
