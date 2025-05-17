import os
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request, send_file, jsonify
from io import BytesIO
import base64
import re
from collections import defaultdict

app = Flask(__name__)

# Ensure the templates directory exists
os.makedirs('templates', exist_ok=True)

# Create basic template
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>AgentXthics Dashboard</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { padding: 20px; }
        .dashboard-card { margin-bottom: 20px; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .chart-container { height: 300px; }
        .log-container { max-height: 400px; overflow-y: auto; font-family: monospace; font-size: 0.85em; }
        .nav-pills .nav-link.active { background-color: #6c757d; }
        pre { white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="my-4">AgentXthics Dashboard</h1>
        
        <ul class="nav nav-pills mb-4">
            <li class="nav-item">
                <a class="nav-link active" href="#overview" data-bs-toggle="tab">Overview</a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="#scenarios" data-bs-toggle="tab">Scenarios</a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="#logs" data-bs-toggle="tab">Logs</a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="#analytics" data-bs-toggle="tab">Analytics</a>
            </li>
        </ul>
        
        <div class="tab-content">
            <!-- Overview Tab -->
            <div class="tab-pane active" id="overview">
                <div class="row">
                    <div class="col-md-6">
                        <div class="dashboard-card bg-light">
                            <h4>Resource Pool Status</h4>
                            <div class="chart-container">
                                <canvas id="resourceChart"></canvas>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="dashboard-card bg-light">
                            <h4>Conservation Rate</h4>
                            <div class="chart-container">
                                <canvas id="conservationChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col-md-6">
                        <div class="dashboard-card bg-light">
                            <h4>Agent Actions</h4>
                            <div class="chart-container">
                                <canvas id="actionsChart"></canvas>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="dashboard-card bg-light">
                            <h4>Communication Network</h4>
                            <div class="chart-container text-center">
                                <img id="networkImage" src="" alt="Communication Network" style="max-height: 280px;">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Scenarios Tab -->
            <div class="tab-pane" id="scenarios">
                <div class="row">
                    <div class="col-md-12 mb-4">
                        <div class="dashboard-card bg-light">
                            <h4>Scenario Comparison</h4>
                            <div class="chart-container">
                                <canvas id="scenarioComparisonChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row" id="scenarioCards">
                    <!-- Scenario cards will be inserted here dynamically -->
                </div>
            </div>
            
            <!-- Logs Tab -->
            <div class="tab-pane" id="logs">
                <div class="row">
                    <div class="col-md-6">
                        <div class="dashboard-card bg-light">
                            <h4>Decision Log</h4>
                            <div class="log-container" id="decisionLog">
                                <pre>Loading decision logs...</pre>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="dashboard-card bg-light">
                            <h4>Message Log</h4>
                            <div class="log-container" id="messageLog">
                                <pre>Loading message logs...</pre>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Analytics Tab -->
            <div class="tab-pane" id="analytics">
                <div class="row">
                    <div class="col-md-6">
                        <div class="dashboard-card bg-light">
                            <h4>Keyword Analysis</h4>
                            <div class="chart-container">
                                <canvas id="keywordChart"></canvas>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="dashboard-card bg-light">
                            <h4>Trust Scores vs. Actions</h4>
                            <div class="chart-container">
                                <canvas id="trustChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row">
                    <div class="col-md-12">
                        <div class="dashboard-card bg-light">
                            <h4>Message Volume by Round</h4>
                            <div class="chart-container">
                                <canvas id="messageVolumeChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Functions to fetch data
        async function fetchData(endpoint) {
            const response = await fetch(endpoint);
            return response.json();
        }
        
        // Initialize charts when data is available
        async function initializeDashboard() {
            const overviewData = await fetchData('/api/overview');
            const scenarioData = await fetchData('/api/scenarios');
            const logData = await fetchData('/api/logs');
            const analyticsData = await fetchData('/api/analytics');
            
            // Update logs
            document.getElementById('decisionLog').innerHTML = `<pre>${logData.decisions}</pre>`;
            document.getElementById('messageLog').innerHTML = `<pre>${logData.messages}</pre>`;
            
            // Network image
            document.getElementById('networkImage').src = 'data:image/png;base64,' + overviewData.network_image;
            
            // Resource pool chart
            new Chart(document.getElementById('resourceChart'), {
                type: 'line',
                data: {
                    labels: overviewData.rounds,
                    datasets: [{
                        label: 'Pool Amount',
                        data: overviewData.pool_amounts,
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Amount'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Round'
                            }
                        }
                    }
                }
            });
            
            // Conservation rate chart
            new Chart(document.getElementById('conservationChart'), {
                type: 'line',
                data: {
                    labels: overviewData.rounds.slice(1),
                    datasets: [{
                        label: 'Conservation Rate',
                        data: overviewData.conservation_rates.slice(1),
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            min: 0,
                            max: 1,
                            title: {
                                display: true,
                                text: 'Rate'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Round'
                            }
                        }
                    }
                }
            });
            
            // Agent actions chart
            new Chart(document.getElementById('actionsChart'), {
                type: 'bar',
                data: {
                    labels: overviewData.agent_ids,
                    datasets: [
                        {
                            label: 'Conserve',
                            data: overviewData.conserve_counts,
                            backgroundColor: 'rgba(75, 192, 192, 0.6)',
                            borderColor: 'rgba(75, 192, 192, 1)',
                            borderWidth: 1
                        },
                        {
                            label: 'Consume',
                            data: overviewData.consume_counts,
                            backgroundColor: 'rgba(255, 99, 132, 0.6)',
                            borderColor: 'rgba(255, 99, 132, 1)',
                            borderWidth: 1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Count'
                            }
                        }
                    }
                }
            });
            
            // Scenario comparison chart
            new Chart(document.getElementById('scenarioComparisonChart'), {
                type: 'line',
                data: {
                    labels: scenarioData.rounds,
                    datasets: scenarioData.datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Pool Amount'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Round'
                            }
                        }
                    }
                }
            });
            
            // Populate scenario cards
            const scenarioCardsContainer = document.getElementById('scenarioCards');
            scenarioData.scenarios.forEach(scenario => {
                const card = document.createElement('div');
                card.className = 'col-md-6 mb-4';
                card.innerHTML = `
                    <div class="dashboard-card bg-light">
                        <h4>${scenario.name}</h4>
                        <div class="text-center">
                            <img src="data:image/png;base64,${scenario.image}" alt="${scenario.name}" style="max-width: 100%; max-height: 200px;">
                        </div>
                        <p class="mt-2"><strong>Final pool amount:</strong> ${scenario.final_amount}</p>
                        <p><strong>Conservation rate:</strong> ${(scenario.conservation_rate * 100).toFixed(1)}%</p>
                    </div>
                `;
                scenarioCardsContainer.appendChild(card);
            });
            
            // Keyword analysis chart
            new Chart(document.getElementById('keywordChart'), {
                type: 'bar',
                data: {
                    labels: analyticsData.keywords,
                    datasets: [
                        {
                            label: 'Frequency',
                            data: analyticsData.keyword_counts,
                            backgroundColor: 'rgba(153, 102, 255, 0.6)',
                            borderColor: 'rgba(153, 102, 255, 1)',
                            borderWidth: 1
                        }
                    ]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Count'
                            }
                        }
                    }
                }
            });
            
            // Trust scores vs actions chart
            new Chart(document.getElementById('trustChart'), {
                type: 'scatter',
                data: {
                    datasets: [
                        {
                            label: 'Conserve',
                            data: analyticsData.trust_conserve,
                            backgroundColor: 'rgba(75, 192, 192, 0.6)',
                            borderColor: 'rgba(75, 192, 192, 1)',
                            pointRadius: 5
                        },
                        {
                            label: 'Consume',
                            data: analyticsData.trust_consume,
                            backgroundColor: 'rgba(255, 99, 132, 0.6)',
                            borderColor: 'rgba(255, 99, 132, 1)',
                            pointRadius: 5
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            title: {
                                display: true,
                                text: 'Action'
                            },
                            type: 'category',
                            labels: ['Consume', 'Conserve']
                        },
                        x: {
                            min: 0,
                            max: 1,
                            title: {
                                display: true,
                                text: 'Trust Score'
                            }
                        }
                    }
                }
            });
            
            // Message volume chart
            new Chart(document.getElementById('messageVolumeChart'), {
                type: 'bar',
                data: {
                    labels: analyticsData.message_rounds,
                    datasets: [{
                        label: 'Message Count',
                        data: analyticsData.message_counts,
                        backgroundColor: 'rgba(255, 159, 64, 0.6)',
                        borderColor: 'rgba(255, 159, 64, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Count'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Round'
                            }
                        }
                    }
                }
            });
        }
        
        // Initialize dashboard on load
        document.addEventListener('DOMContentLoaded', initializeDashboard);
    </script>
</body>
</html>
''')

# Class to process and analyze log data
class LogAnalyzer:
    def __init__(self, base_dir='agentxthics'):
        self.base_dir = base_dir
        
    def parse_decision_log(self, file_path):
        """Parse decision log into a structured format."""
        decisions = []
        
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(', ', 3)
                if len(parts) < 4:
                    continue
                    
                round_num, agent_id, action, explanation = parts
                
                decisions.append({
                    'round': int(round_num),
                    'agent': agent_id,
                    'action': action,
                    'explanation': explanation
                })
        
        return decisions
    
    def parse_message_log(self, file_path):
        """Parse message log into a structured format."""
        messages = []
        
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(', ', 3)
                if len(parts) < 3:
                    continue
                
                if len(parts) == 3:
                    round_num, sender, recipient = parts
                    content = ""
                else:
                    round_num, sender, recipient, content = parts
                
                messages.append({
                    'round': int(round_num),
                    'sender': sender,
                    'recipient': recipient,
                    'content': content
                })
        
        return messages
    
    def parse_state_log(self, file_path):
        """Parse state log JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def get_agent_action_counts(self, decisions):
        """Count conserve/consume actions by agent."""
        agent_actions = defaultdict(lambda: {'conserve': 0, 'consume': 0})
        
        for decision in decisions:
            agent = decision['agent']
            action = decision['action']
            
            if action == 'conserve':
                agent_actions[agent]['conserve'] += 1
            elif action == 'consume':
                agent_actions[agent]['consume'] += 1
        
        return agent_actions
    
    def get_trust_mentions(self, decisions):
        """Extract trust scores mentioned in explanations."""
        trust_pattern = r'trust score[s]* (?:for|of|is|are)? ?(?:\w+)?:? ?([-+]?[0-9]*\.?[0-9]+)'
        trust_mentions = []
        
        for decision in decisions:
            explanation = decision['explanation'].lower()
            
            if 'trust' in explanation:
                matches = re.findall(trust_pattern, explanation)
                
                for score in matches:
                    try:
                        trust_mentions.append({
                            'round': decision['round'],
                            'agent': decision['agent'],
                            'trust_score': float(score),
                            'action': decision['action']
                        })
                    except ValueError:
                        continue
        
        return trust_mentions
    
    def get_keyword_counts(self, messages):
        """Count keyword occurrences in messages."""
        keywords = {
            'cooperation': ['cooperate', 'collaborate', 'together', 'all'],
            'benefit': ['benefit', 'gain', 'profit', 'advantage'],
            'trust': ['trust', 'reliable', 'honest', 'believe'],
            'refill': ['refill', 'replenish', 'renew', 'restore'],
            'agreement': ['agree', 'yes', 'okay', 'sure'],
            'disagreement': ['disagree', 'no', 'not', 'don\'t'],
            'strategy': ['strategy', 'plan', 'approach', 'method']
        }
        
        keyword_counts = defaultdict(int)
        
        for message in messages:
            content = message['content'].lower()
            
            for category, words in keywords.items():
                for word in words:
                    if re.search(r'\b' + word + r'\b', content):
                        keyword_counts[category] += 1
                        break  # Count only once per category per message
        
        return keyword_counts
    
    def generate_network_image(self, messages):
        """Generate a communication network visualization."""
        plt.figure(figsize=(8, 6))
        
        # Count message frequencies between agents
        edges = defaultdict(int)
        nodes = set()
        
        for message in messages:
            sender = message['sender']
            recipient = message['recipient']
            
            nodes.add(sender)
            nodes.add(recipient)
            edges[(sender, recipient)] += 1
        
        # Create positions for nodes
        positions = {}
        num_nodes = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            angle = 2 * np.pi * i / num_nodes
            positions[node] = (np.cos(angle), np.sin(angle))
        
        # Draw nodes
        for node, pos in positions.items():
            plt.scatter(pos[0], pos[1], s=300, c='lightblue', edgecolors='black', zorder=2)
            plt.text(pos[0], pos[1], node, ha='center', va='center', fontweight='bold')
        
        # Draw edges
        max_weight = max(edges.values()) if edges else 1
        for (sender, recipient), weight in edges.items():
            start_pos = positions[sender]
            end_pos = positions[recipient]
            
            # Calculate arrow curve
            # If bidirectional, curve both edges
            is_bidirectional = (recipient, sender) in edges
            curve = 0.2 if is_bidirectional else 0
            
            # Calculate control point for curved arrow
            mid_x = (start_pos[0] + end_pos[0]) / 2
            mid_y = (start_pos[1] + end_pos[1]) / 2
            
            # Perpendicular direction
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            
            # Calculate normal vector
            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                normal_x = -dy / length
                normal_y = dx / length
            else:
                normal_x, normal_y = 0, 0
            
            # Control point
            ctrl_x = mid_x + curve * normal_x
            ctrl_y = mid_y + curve * normal_y
            
            # Draw curved arrow
            arrow_width = 1 + 3 * (weight / max_weight)
            arrow_color = plt.cm.Blues(0.5 + 0.5 * weight / max_weight)
            
            plt.annotate(
                '',
                xy=end_pos, xycoords='data',
                xytext=start_pos, textcoords='data',
                arrowprops=dict(
                    arrowstyle='-|>',
                    connectionstyle=f'arc3,rad={curve}',
                    color=arrow_color,
                    lw=arrow_width,
                    alpha=0.7
                ),
                zorder=1
            )
            
            # Add weight label
            if weight > 1:
                label_x = ctrl_x
                label_y = ctrl_y
                plt.text(label_x, label_y, str(weight), 
                        ha='center', va='center', 
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.7))
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save to BytesIO
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return image_base64

# API endpoints
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/overview')
def api_overview():
    analyzer = LogAnalyzer()
    
    # Check for enhanced simulation results first
    if os.path.exists('state_log.json'):
        state_log = analyzer.parse_state_log('state_log.json')
        decisions = analyzer.parse_decision_log('enhanced_decision_log.csv')
        messages = analyzer.parse_message_log('enhanced_message_log.csv')
    else:
        # Fall back to original logs
        decisions = analyzer.parse_decision_log('decision_log.csv')
        messages = analyzer.parse_message_log('message_log.csv')
        
        # Create mock state log from decisions
        state_log = []
        rounds = sorted(set(d['round'] for d in decisions))
        for r in rounds:
            round_decisions = [d for d in decisions if d['round'] == r]
            conserve_count = sum(1 for d in round_decisions if d['action'] == 'conserve')
            consume_count = sum(1 for d in round_decisions if d['action'] == 'consume')
            total_count = conserve_count + consume_count
            
            state_log.append({
                'round': r,
                'conservation_rate': conserve_count / total_count if total_count > 0 else 0
            })
    
    # Extract data for charts
    agent_action_counts = analyzer.get_agent_action_counts(decisions)
    network_image = analyzer.generate_network_image(messages)
    
    # Prepare data for charts
    rounds = [state['round'] for state in state_log]
    pool_amounts = [state.get('amount', 50) for state in state_log]  # Default to 50 if not present
    conservation_rates = [state.get('conservation_rate', 0) for state in state_log]
    
    agent_ids = sorted(agent_action_counts.keys())
    conserve_counts = [agent_action_counts[agent]['conserve'] for agent in agent_ids]
    consume_counts = [agent_action_counts[agent]['consume'] for agent in agent_ids]
    
    return jsonify({
        'rounds': rounds,
        'pool_amounts': pool_amounts,
        'conservation_rates': conservation_rates,
        'agent_ids': agent_ids,
        'conserve_counts': conserve_counts,
        'consume_counts': consume_counts,
        'network_image': network_image
    })

@app.route('/api/scenarios')
def api_scenarios():
    # Check for scenario results
    scenarios = []
    scenario_datasets = []
    max_rounds = 0
    
    # Check for cooperative scenario
    if os.path.exists('cooperative_scenario/state_log.json'):
        with open('cooperative_scenario/state_log.json', 'r') as f:
            coop_data = json.load(f)
            
        coop_rounds = [state['round'] for state in coop_data]
        coop_amounts = [state.get('amount', 50) for state in coop_data]
        max_rounds = max(max_rounds, len(coop_rounds))
        
        with open('cooperative_scenario/simulation_results.png', 'rb') as f:
            coop_image = base64.b64encode(f.read()).decode('utf-8')
        
        # Calculate conservation rate
        coop_conserve_rate = sum(1 for state in coop_data if state.get('conservation_rate', 0) >= 0.5) / len(coop_data)
        
        scenarios.append({
            'name': 'Cooperative Scenario',
            'image': coop_image,
            'final_amount': coop_amounts[-1],
            'conservation_rate': coop_conserve_rate
        })
        
        scenario_datasets.append({
            'label': 'Cooperative',
            'data': coop_amounts,
            'borderColor': 'rgba(75, 192, 192, 1)',
            'backgroundColor': 'rgba(75, 192, 192, 0.2)',
            'fill': False
        })
    
    # Check for competitive scenario
    if os.path.exists('competitive_scenario/state_log.json'):
        with open('competitive_scenario/state_log.json', 'r') as f:
            comp_data = json.load(f)
            
        comp_rounds = [state['round'] for state in comp_data]
        comp_amounts = [state.get('amount', 50) for state in comp_data]
        max_rounds = max(max_rounds, len(comp_rounds))
        
        with open('competitive_scenario/simulation_results.png', 'rb') as f:
            comp_image = base64.b64encode(f.read()).decode('utf-8')
        
        # Calculate conservation rate
        comp_conserve_rate = sum(1 for state in comp_data if state.get('conservation_rate', 0) >= 0.5) / len(comp_data)
        
        scenarios.append({
            'name': 'Competitive Scenario',
            'image': comp_image,
            'final_amount': comp_amounts[-1],
            'conservation_rate': comp_conserve_rate
        })
        
        scenario_datasets.append({
            'label': 'Competitive',
            'data': comp_amounts,
            'borderColor': 'rgba(255, 99, 132, 1)',
            'backgroundColor': 'rgba(255, 99, 132, 0.2)',
            'fill': False
        })
    
    # Check for mixed scenario
    if os.path.exists('mixed_scenario/state_log.json'):
        with open('mixed_scenario/state_log.json', 'r') as f:
            mixed_data = json.load(f)
            
        mixed_rounds = [state['round'] for state in mixed_data]
        mixed_amounts = [state.get('amount', 50) for state in mixed_data]
        max_rounds = max(max_rounds, len(mixed_rounds))
        
        with open('mixed_scenario/simulation_results.png', 'rb') as f:
            mixed_image = base64.b64encode(f.read()).decode('utf-8')
        
        # Calculate conservation rate
        mixed_conserve_rate = sum(1 for state in mixed_data if state.get('conservation_rate', 0) >= 0.5) / len(mixed_data)
        
        scenarios.append({
            'name': 'Mixed Agents Scenario',
            'image': mixed_image,
            'final_amount': mixed_amounts[-1],
            'conservation_rate': mixed_conserve_rate
        })
        
        scenario_datasets.append({
            'label': 'Mixed Agents',
            'data': mixed_amounts,
            'borderColor': 'rgba(54, 162, 235, 1)',
            'backgroundColor': 'rgba(54, 162, 235, 0.2)',
            'fill': False
        })
    
    # Check for tragedy scenario
    if os.path.exists('tragedy_scenario/state_log.json'):
        with open('tragedy_scenario/state_log.json', 'r') as f:
            tragedy_data = json.load(f)
            
        tragedy_rounds = [state['round'] for state in tragedy_data]
        tragedy_amounts = [state.get('amount', 50) for state in tragedy_data]
        max_rounds = max(max_rounds, len(tragedy_rounds))
        
        with open('tragedy_scenario/simulation_results.png', 'rb') as f:
            tragedy_image = base64.b64encode(f.read()).decode('utf-8')
        
        # Calculate conservation rate
        tragedy_conserve_rate = sum(1 for state in tragedy_data if state.get('conservation_rate', 0) >= 0.5) / len(tragedy_data)
        
        scenarios.append({
            'name': 'Tragedy of the Commons Scenario',
            'image': tragedy_image,
            'final_amount': tragedy_amounts[-1],
            'conservation_rate': tragedy_conserve_rate
        })
        
        scenario_datasets.append({
            'label': 'Tragedy of the Commons',
            'data': tragedy_amounts,
            'borderColor': 'rgba(255, 159, 64, 1)',
            'backgroundColor': 'rgba(255, 159, 64, 0.2)',
            'fill': False
        })
    
    # If no scenarios are available, provide a default dataset
    if not scenarios:
        rounds = list(range(11))  # 0-10
        default_data = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]  # Example data
        
        scenarios.append({
            'name': 'Default Scenario',
            'image': '',
            'final_amount': default_data[-1],
            'conservation_rate': 1.0
        })
        
        scenario_datasets.append({
            'label': 'Default',
            'data': default_data,
            'borderColor': 'rgba(153, 102, 255, 1)',
            'backgroundColor': 'rgba(153, 102, 255, 0.2)',
            'fill': False
        })
        
        max_rounds = 11
    
    return jsonify({
        'rounds': list(range(max_rounds)),
        'datasets': scenario_datasets,
        'scenarios': scenarios
    })

@app.route('/api/logs')
def api_logs():
    # Select the appropriate log files
    decision_log_path = 'enhanced_decision_log.csv'
    message_log_path = 'enhanced_message_log.csv'
    
    if not os.path.exists(decision_log_path):
        decision_log_path = 'decision_log.csv'
    
    if not os.path.exists(message_log_path):
        message_log_path = 'message_log.csv'
    
    # Read log files
    decisions = ""
    messages = ""
    
    try:
        with open(decision_log_path, 'r') as f:
            decisions = f.read()
    except Exception as e:
        decisions = f"Error reading decision log: {str(e)}"
    
    try:
        with open(message_log_path, 'r') as f:
            messages = f.read()
    except Exception as e:
        messages = f"Error reading message log: {str(e)}"
    
    return jsonify({
        'decisions': decisions,
        'messages': messages
    })

@app.route('/api/analytics')
def api_analytics():
    analyzer = LogAnalyzer()
    
    # Check for enhanced simulation results first
    decision_log_path = 'enhanced_decision_log.csv'
    message_log_path = 'enhanced_message_log.csv'
    
    if not os.path.exists(decision_log_path):
        decision_log_path = 'decision_log.csv'
    
    if not os.path.exists(message_log_path):
        message_log_path = 'message_log.csv'
    
    decisions = analyzer.parse_decision_log(decision_log_path)
    messages = analyzer.parse_message_log(message_log_path)
    
    # Keyword analysis
    keyword_counts = analyzer.get_keyword_counts(messages)
    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
    keywords = [k for k, _ in sorted_keywords]
    counts = [c for _, c in sorted_keywords]
    
    # Trust score analysis
    trust_mentions = analyzer.get_trust_mentions(decisions)
    
    trust_conserve = [
        {'x': mention['trust_score'], 'y': 1}  # y=1 for conserve
        for mention in trust_mentions if mention['action'] == 'conserve'
    ]
    trust_consume = [
        {'x': mention['trust_score'], 'y': 0}  # y=0 for consume
        for mention in trust_mentions if mention['action'] == 'consume'
    ]
    
    # Message volume by round
    message_counts = defaultdict(int)
    for message in messages:
        message_counts[message['round']] += 1
    
    message_rounds = sorted(message_counts.keys())
    message_count_values = [message_counts[r] for r in message_rounds]
    
    return jsonify({
        'keywords': keywords,
        'keyword_counts': counts,
        'trust_conserve': trust_conserve,
        'trust_consume': trust_consume,
        'message_rounds': message_rounds,
        'message_counts': message_count_values
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
