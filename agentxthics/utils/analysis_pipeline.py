"""
Analysis Pipeline for AgentXthics.
This module handles the automatic analysis of simulation logs and generates visualizations.
"""
import os
import sys
import json
import time
import datetime
import subprocess
import shutil
from pathlib import Path

def run_analysis_pipeline(simulation_result, output_dir, config_file=None):
    """
    Run the complete analysis pipeline on simulation results.
    
    Args:
        simulation_result: The result object from the simulation
        output_dir: Directory where simulation logs are stored
        config_file: Path to the configuration file used for the simulation
    
    Returns:
        Dictionary with analysis results and paths to visualization files
    """
    print("\n=== Running Analysis Pipeline ===")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Analysis started at: {timestamp}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save basic simulation data
    save_simulation_data(simulation_result, output_dir)
    
    # Run log analysis
    analyze_logs(output_dir)
    
    # Run decision analysis
    # Create CSV version of decision log for the analyzer
    decision_log_json = os.path.join(output_dir, "decision_log.json")
    decision_log_csv = os.path.join(output_dir, "decision_log.csv")
    
    # Initialize result variables
    decision_analysis_results = {"status": "unknown", "output_dir": output_dir}
    
    print(f"Converting decision log from JSON to CSV...")
    
    # Check if JSON exists before attempting conversion
    if os.path.exists(decision_log_json):
        try:
            with open(decision_log_json, 'r') as f:
                decisions = json.load(f)
            
            # Create CSV with header
            with open(decision_log_csv, 'w') as f:
                f.write("round, agent_id, action, explanation\n")  # Add header
                
                # Process each decision
                for decision in decisions:
                    if isinstance(decision, dict):
                        # Handle dict format
                        round_num = decision.get('round', '')
                        agent_id = decision.get('agent_id', '')
                        action = decision.get('action', '')
                        explanation = str(decision.get('explanation', '')).replace(',', ' ')
                        f.write(f"{round_num}, {agent_id}, {action}, {explanation}\n")
                    elif isinstance(decision, list) and len(decision) >= 4:
                        # Handle list format
                        round_num, agent_id, action, explanation = decision[:4]
                        explanation = str(explanation).replace(',', ' ')
                        f.write(f"{round_num}, {agent_id}, {action}, {explanation}\n")
            
            print(f"Successfully converted decision log to CSV at {decision_log_csv}")
        except Exception as e:
            print(f"Error converting decision log to CSV: {e}")
            # Create minimal file to prevent analysis failures
            with open(decision_log_csv, 'w') as f:
                f.write("round, agent_id, action, explanation\n")
    else:
        print(f"WARNING: decision_log.json not found at {decision_log_json}")
        # Create minimal file to prevent analysis failures
        with open(decision_log_csv, 'w') as f:
            f.write("round, agent_id, action, explanation\n")
        
        try:
            # Import the analyze_decisions module
            from agentxthics.utils.analyze_decisions import main as analyze_decisions_main
            
            # Run the analysis with the correct directory
            print("Running decision analysis...")
            cwd = os.getcwd()
            os.chdir(output_dir)  # Move to the output directory to save files there
            analyze_decisions_main(output_dir)  # Pass the directory
            os.chdir(cwd)  # Restore original directory
            
            # Check if visualization was created
            if os.path.exists(os.path.join(output_dir, "decision_analysis.png")):
                decision_analysis_results = {"status": "success", "output_dir": output_dir}
                print("Decision analysis completed successfully")
            else:
                decision_analysis_results = {"status": "error", "output_dir": output_dir}
                print("Decision analysis did not generate visualization")
        except Exception as e:
            print(f"Error in decision analysis: {e}")
            decision_analysis_results = {"status": "error", "error": str(e)}
    
    # Run message analysis
    # Create CSV version of message log for the analyzer
    message_log_json = os.path.join(output_dir, "message_log.json")
    message_log_csv = os.path.join(output_dir, "message_log.csv")
    
    print(f"Converting message log from JSON to CSV...")
    
    # Check if JSON exists before attempting conversion
    if os.path.exists(message_log_json):
        try:
            with open(message_log_json, 'r') as f:
                messages = json.load(f)
            
            # Create CSV with header
            with open(message_log_csv, 'w') as f:
                f.write("round, sender, recipient, message\n")  # Add header
                
                # Process each message
                for message in messages:
                    if isinstance(message, dict):
                        # Handle dict format
                        round_num = message.get('round', '')
                        sender = message.get('sender_id', '') or message.get('sender', '')
                        recipient = message.get('recipient_id', '') or message.get('recipient', '')
                        content = message.get('message', '') or message.get('content', '')
                        # Escape commas in content
                        content = str(content).replace(',', '\\,')
                        f.write(f"{round_num}, {sender}, {recipient}, {content}\n")
                    elif isinstance(message, list) and len(message) >= 3:
                        # Handle list format
                        if len(message) >= 4:
                            round_num, sender, recipient, content = message[:4]
                        else:
                            round_num, sender, recipient = message[:3]
                            content = ""
                        # Escape commas in content
                        content = str(content).replace(',', '\\,')
                        f.write(f"{round_num}, {sender}, {recipient}, {content}\n")
            
            print(f"Successfully converted message log to CSV at {message_log_csv}")
        except Exception as e:
            print(f"Error converting message log to CSV: {e}")
            # Create minimal file to prevent analysis failures
            with open(message_log_csv, 'w') as f:
                f.write("round, sender, recipient, message\n")
    else:
        print(f"WARNING: message_log.json not found at {message_log_json}")
        # Create minimal file to prevent analysis failures
        with open(message_log_csv, 'w') as f:
            f.write("round, sender, recipient, message\n")

    # Verify if message_log.csv was created successfully
    if os.path.exists(message_log_csv):
        print(f"Confirmed message_log.csv exists at: {message_log_csv}")
        with open(message_log_csv, 'r') as f:
            first_line = f.readline().strip()
            print(f"First line of message_log.csv: {first_line}")
    else:
        print(f"WARNING: message_log.csv creation failed!")
    
    try:
        # Import the analyze_messages module
        from agentxthics.utils.analyze_messages import main as analyze_messages_main
        
        # Run the analysis with the correct directory
        print("Running message analysis...")
        cwd = os.getcwd()
        # Don't change directory, but pass the absolute path to the output directory
        analyze_messages_main(os.path.abspath(output_dir))
        
        # Check if visualization was created
        if os.path.exists(os.path.join(output_dir, "message_analysis.png")):
            message_analysis_results = {"status": "success", "output_dir": output_dir}
            print("Message analysis completed successfully")
        else:
            message_analysis_results = {"status": "error", "output_dir": output_dir}
            print("Message analysis did not generate visualization")
    except Exception as e:
        print(f"Error in message analysis: {e}")
        message_analysis_results = {"status": "error", "error": str(e)}
    
    # Generate dashboard
    dashboard_url = generate_dashboard(output_dir)
    
    # Add a timestamp to the summary file
    add_timestamp_to_summary(output_dir, timestamp)
    
    print(f"Analysis completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {output_dir}")
    
    # Optionally open dashboard if it was generated
    if dashboard_url:
        print(f"Dashboard available at: {dashboard_url}")
    
    return {
        "output_dir": output_dir,
        "timestamp": timestamp,
        "decision_analysis": decision_analysis_results,
        "message_analysis": message_analysis_results,
        "dashboard_url": dashboard_url
    }

def save_simulation_data(simulation_result, output_dir):
    """
    Save the raw simulation data to JSON files.
    
    Args:
        simulation_result: The result object from the simulation
        output_dir: Directory where to save the data
    """
    # Extract data from simulation result
    try:
        # Save state log
        state_log_path = os.path.join(output_dir, "state_log.json")
        with open(state_log_path, 'w') as f:
            json.dump(simulation_result.state_log, f, indent=2)
        
        # Save decision log
        decision_log_path = os.path.join(output_dir, "decision_log.json")
        with open(decision_log_path, 'w') as f:
            json.dump(simulation_result.decision_log, f, indent=2)
        
        # Save message log if available
        if hasattr(simulation_result, 'message_log'):
            message_log_path = os.path.join(output_dir, "message_log.json")
            with open(message_log_path, 'w') as f:
                json.dump(simulation_result.message_log, f, indent=2)
        
        # Save timeout log if available
        timeout_data = []
        if hasattr(simulation_result, 'shock_history'):
            timeout_log_path = os.path.join(output_dir, "timeout_log.json")
            with open(timeout_log_path, 'w') as f:
                json.dump(timeout_data, f, indent=2)
        
        print(f"Saved raw simulation data to {output_dir}")
    except Exception as e:
        print(f"Error saving simulation data: {e}")

def analyze_logs(log_dir):
    """
    Run the log analysis script on the simulation logs.
    
    Args:
        log_dir: Directory containing the simulation logs
    
    Returns:
        Dictionary with analysis results
    """
    try:
        # Import the analyze_logs module
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from agentxthics.utils.analyze_logs import main as analyze_logs_main
        
        # Run the analysis
        print("Running log analysis...")
        analyze_logs_main(log_dir)
        
        # Check if analysis files were created
        expected_files = ['agent_conversations.txt', 'agent_decisions.txt', 'analysis_summary.txt']
        all_created = all(os.path.exists(os.path.join(log_dir, file)) for file in expected_files)
        
        if all_created:
            print("Log analysis completed successfully")
            return {"status": "success", "output_dir": log_dir}
        else:
            missing = [f for f in expected_files if not os.path.exists(os.path.join(log_dir, f))]
            print(f"Log analysis partial: missing files {missing}")
            return {"status": "partial", "missing_files": missing}
    except Exception as e:
        print(f"Error during log analysis: {e}")
        return {"status": "error", "error": str(e)}

def analyze_decisions(log_dir):
    """
    Run the decision analysis script on the decision logs.
    
    Args:
        log_dir: Directory containing the decision logs
    
    Returns:
        Dictionary with analysis results
    """
    try:
        # Create CSV version of decision log for the analyzer
        decision_log_json = os.path.join(log_dir, "decision_log.json")
        decision_log_csv = os.path.join(log_dir, "decision_log.csv")
        
        if os.path.exists(decision_log_json):
            # Convert JSON to CSV format expected by analyze_decisions
            with open(decision_log_json, 'r') as f:
                decisions = json.load(f)
            
            with open(decision_log_csv, 'w') as f:
                for decision in decisions:
                    if isinstance(decision, list) and len(decision) >= 4:
                        f.write(f"{decision[0]}, {decision[1]}, {decision[2]}, {decision[3]}\n")
        
        # Import the analyze_decisions module
        from agentxthics.utils.analyze_decisions import main as analyze_decisions_main
        
        # Run analysis from the log directory
        cwd = os.getcwd()
        os.chdir(log_dir)
        
        # Run the analysis
        print("Running decision analysis...")
        analyze_decisions_main()
        
        # Restore working directory
        os.chdir(cwd)
        
        # Check if analysis visualization was created
        visualization_path = os.path.join(log_dir, "decision_analysis.png")
        if os.path.exists(visualization_path):
            print("Decision analysis completed successfully")
            return {"status": "success", "visualization": visualization_path}
        else:
            print("Decision analysis did not generate visualization")
            return {"status": "error", "error": "No visualization generated"}
    except Exception as e:
        print(f"Error during decision analysis: {e}")
        return {"status": "error", "error": str(e)}

def analyze_messages(log_dir):
    """
    Run the message analysis script on the message logs.
    
    Args:
        log_dir: Directory containing the message logs
    
    Returns:
        Dictionary with analysis results
    """
    try:
        # Create CSV version of message log for the analyzer
        message_log_json = os.path.join(log_dir, "message_log.json")
        message_log_csv = os.path.join(log_dir, "message_log.csv")
        
        # Check if message log exists and save it to CSV
        if os.path.exists(message_log_json):
            # Convert JSON to CSV format expected by analyze_messages
            try:
                with open(message_log_json, 'r') as f:
                    messages = json.load(f)
                
                # Ensure file exists even if empty - create with header
                with open(message_log_csv, 'w') as f:
                    f.write("round, sender, recipient, message\n")  # Add header
                    
                    # Check if messages is empty
                    if not messages:
                        print("WARNING: message_log.json exists but contains no messages")
                        print("This could be because agents did not communicate during the simulation")
                    
                    # Process any messages
                    for message in messages:
                        if isinstance(message, list):
                            if len(message) >= 4:
                                # Escape any commas in the message content
                                round_num, sender, recipient, content = message
                                content = content.replace(',', '\\,') if isinstance(content, str) else content
                                f.write(f"{round_num}, {sender}, {recipient}, {content}\n")
                            elif len(message) == 3:
                                round_num, sender, recipient = message
                                f.write(f"{round_num}, {sender}, {recipient}, \n")
                        elif isinstance(message, dict):
                            # Handle dict format if it exists
                            round_num = message.get('round', '')
                            sender = message.get('sender', '')
                            recipient = message.get('recipient', '')
                            content = message.get('content', '').replace(',', '\\,') if isinstance(message.get('content', ''), str) else ''
                            f.write(f"{round_num}, {sender}, {recipient}, {content}\n")
            except Exception as e:
                print(f"ERROR converting message log to CSV: {e}")
                # Create minimal file to prevent analysis failures
                with open(message_log_csv, 'w') as f:
                    f.write("round, sender, recipient, message\n")
        else:
            print("WARNING: message_log.json does not exist. Creating empty message_log.csv")
            with open(message_log_csv, 'w') as f:
                f.write("round, sender, recipient, message\n")
        
        # Import the analyze_messages module
        from agentxthics.utils.analyze_messages import main as analyze_messages_main
        
        # Run analysis from the log directory
        cwd = os.getcwd()
        os.chdir(log_dir)
        
        # Run the analysis
        print("Running message analysis...")
        analyze_messages_main()
        
        # Restore working directory
        os.chdir(cwd)
        
        # Check if analysis visualization was created
        visualization_path = os.path.join(log_dir, "message_analysis.png")
        if os.path.exists(visualization_path):
            print("Message analysis completed successfully")
            return {"status": "success", "visualization": visualization_path}
        else:
            print("Message analysis did not generate visualization")
            return {"status": "error", "error": "No visualization generated"}
    except Exception as e:
        print(f"Error during message analysis: {e}")
        return {"status": "error", "error": str(e)}

def generate_dashboard(log_dir):
    """
    Generate and start the visualization dashboard.
    
    Args:
        log_dir: Directory containing the analysis results
    
    Returns:
        URL to the dashboard if successful, None otherwise
    """
    try:
        # Copy necessary files for dashboard
        for file in ["decision_log.csv", "message_log.csv", "decision_analysis.png", "message_analysis.png"]:
            source = os.path.join(log_dir, file)
            if os.path.exists(source):
                shutil.copy(source, ".")
        
        # Create templates directory in the output directory
        templates_dir = os.path.join(log_dir, "templates")
        os.makedirs(templates_dir, exist_ok=True)
        
        # Write a more robust dashboard launcher script
        dashboard_script = os.path.join(log_dir, "launch_dashboard.py")
        with open(dashboard_script, 'w') as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
Dashboard launcher for AgentXthics visualization.
\"\"\"
import os
import sys
import importlib.util

# Add the project root directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

# First try to import directly (if package is installed)
try:
    from agentxthics.visualization.dashboard import app
    print("Imported dashboard from installed package")
except ImportError:
    # If that fails, try to import using the relative path
    try:
        # Check if the dashboard.py file exists in the expected location
        dashboard_path = os.path.join(project_root, 'agentxthics', 'visualization', 'dashboard.py')
        
        if os.path.exists(dashboard_path):
            # Dynamically import the module
            spec = importlib.util.spec_from_file_location('dashboard', dashboard_path)
            dashboard_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dashboard_module)
            app = dashboard_module.app
            print(f"Imported dashboard from: {dashboard_path}")
        else:
            # Copy the dashboard file directly to this directory as a fallback
            print("Dashboard module not found at expected location. Using local copy.")
            from dashboard_local import app
    except Exception as e:
        print(f"Error importing dashboard: {e}")
        print("You may need to install the agentxthics package using:")
        print("  pip install -e .")
        print("Or ensure the dashboard.py file is available in agentxthics/visualization/")
        sys.exit(1)

# Make visualization files available locally
try:
    # Copy necessary dashboard data to local directory
    for file in ["decision_log.csv", "message_log.csv"]:
        if not os.path.exists(os.path.join(script_dir, file)) and os.path.exists(os.path.join(project_root, file)):
            import shutil
            shutil.copy(os.path.join(project_root, file), script_dir)
            print(f"Copied {file} to dashboard directory")
except Exception as e:
    print(f"Warning: Could not copy data files: {e}")

# Start the Flask application
print("\\nStarting dashboard server at http://localhost:5000")
print("Press Ctrl+C to exit\\n")
app.run(debug=False, port=5000, host='127.0.0.1')
""")
        
        # Create a local fallback version of the dashboard
        dashboard_local = os.path.join(log_dir, "dashboard_local.py") 
        with open(dashboard_local, 'w') as f:
            f.write("""#!/usr/bin/env python3
\"\"\"
Local copy of the dashboard visualization for AgentXthics.
This file is a fallback if the main dashboard module cannot be imported.
\"\"\"
import sys
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

# Create templates directory
os.makedirs('templates', exist_ok=True)

# Create basic template
basic_template = '''
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
        
        <div class="alert alert-info">
            <strong>Fallback Mode:</strong> Using local dashboard implementation. Some features may be limited.
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="dashboard-card bg-light">
                    <h4>Decision Analysis</h4>
                    <div class="chart-container text-center">
                        <img id="decisionImage" src="/decision-image" alt="Decision Analysis" style="max-height: 280px;">
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="dashboard-card bg-light">
                    <h4>Message Analysis</h4>
                    <div class="chart-container text-center">
                        <img id="messageImage" src="/message-image" alt="Message Analysis" style="max-height: 280px;">
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="dashboard-card bg-light">
                    <h4>Decision Log</h4>
                    <div class="log-container" id="decisionLog">
                        <pre>{{decision_log|safe}}</pre>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="dashboard-card bg-light">
                    <h4>Message Log</h4>
                    <div class="log-container" id="messageLog">
                        <pre>{{message_log|safe}}</pre>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

with open('templates/index.html', 'w') as f:
    f.write(basic_template)

@app.route('/')
def index():
    # Read logs
    decision_log = ""
    message_log = ""
    
    try:
        with open('agent_decisions.txt', 'r') as f:
            decision_log = f.read()
    except:
        try:
            # Try in parent directory
            with open('../agent_decisions.txt', 'r') as f:
                decision_log = f.read()
        except:
            decision_log = "Decision log not found"
    
    try:
        with open('agent_conversations.txt', 'r') as f:
            message_log = f.read()
    except:
        try:
            # Try in parent directory
            with open('../agent_conversations.txt', 'r') as f:
                message_log = f.read()
        except:
            message_log = "Message log not found"
    
    return render_template('index.html', decision_log=decision_log, message_log=message_log)

@app.route('/decision-image')
def decision_image():
    # Check for existing decision analysis image
    for image_path in ['decision_analysis.png', '../decision_analysis.png']:
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                return send_file(BytesIO(f.read()), mimetype='image/png')
    
    # Generate a simple placeholder if no image found
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.5, "Decision analysis image not found", 
             ha='center', va='center', fontsize=14)
    plt.axis('off')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    return send_file(buffer, mimetype='image/png')

@app.route('/message-image')
def message_image():
    # Check for existing message analysis image
    for image_path in ['message_analysis.png', '../message_analysis.png']:
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                return send_file(BytesIO(f.read()), mimetype='image/png')
    
    # Generate a simple placeholder if no image found
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.5, "Message analysis image not found", 
             ha='center', va='center', fontsize=14)
    plt.axis('off')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    return send_file(buffer, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
""")
        
        print("Dashboard files prepared")
        print("To view the dashboard, run:")
        print(f"python {dashboard_script}")
        
        # Create a static HTML dashboard for direct browser viewing
        dashboard_static = os.path.join(log_dir, "dashboard_static.html")
        with open(dashboard_static, 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>AgentXthics Static Dashboard</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 { 
            color: #2c3e50; 
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        h2 { 
            color: #3498db; 
            margin-top: 30px;
        }
        .dashboard-section {
            margin-bottom: 40px;
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .image-container {
            text-align: center;
            margin: 20px 0;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }
        .note {
            background-color: #e7f3fe;
            border-left: 6px solid #2196F3;
            padding: 10px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>AgentXthics Dashboard (Static Version)</h1>
        
        <div class="note">
            <p><strong>Note:</strong> This is a static version of the dashboard that doesn't require a Flask server. 
            For the interactive version, run the launch_dashboard.py script.</p>
        </div>
        
        <div class="dashboard-section">
            <h2>Decision Analysis</h2>
            <div class="image-container">
                <img src="decision_analysis.png" alt="Decision Analysis">
            </div>
        </div>
        
        <div class="dashboard-section">
            <h2>Message Analysis</h2>
            <div class="image-container">
                <img src="message_analysis.png" alt="Message Analysis">
            </div>
        </div>
        
        <div class="dashboard-section">
            <h2>Raw Data Files</h2>
            <p>The following files contain the raw data from the simulation:</p>
            <ul>
                <li>state_log.json - Resource state by round</li>
                <li>decision_log.json - Agent decisions</li>
                <li>message_log.json - Agent communications</li>
                <li>timeout_log.json - Timeout events</li>
            </ul>
        </div>
    </div>
</body>
</html>""")
        
        # Make dashboard script executable on Unix-like systems
        try:
            import stat
            st = os.stat(dashboard_script)
            os.chmod(dashboard_script, st.st_mode | stat.S_IEXEC)
            print("Made dashboard launcher executable")
        except:
            pass
        
        print("\nDashboard options:")
        print(f"1. Dynamic dashboard: python {dashboard_script}")
        print(f"2. Static dashboard: open {dashboard_static} in your browser")
        
        # Return the URL (not starting the server automatically to avoid blocking)
        return "http://localhost:8080"
    except Exception as e:
        print(f"Error preparing dashboard: {e}")
        return None

def add_timestamp_to_summary(output_dir, timestamp):
    """
    Add a timestamp to the analysis summary file.
    
    Args:
        output_dir: Directory containing the analysis summary
        timestamp: Timestamp to add
    """
    summary_path = os.path.join(output_dir, "analysis_summary.txt")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            content = f.read()
        
        # Add timestamp at the top of the file
        with open(summary_path, 'w') as f:
            f.write(f"Analysis generated at: {timestamp}\n\n")
            f.write(content)
        
        print(f"Added timestamp to analysis summary")
