"""Extremely minimal test script for AgentXthics"""
import os
import sys

print("Python version:", sys.version)
print("\nCurrent directory:", os.getcwd())
print("\nListing all Python files in the current directory:")

# List all Python files in the current directory
python_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            python_files.append(os.path.join(root, file))

for file in sorted(python_files):
    print(f"- {file}")

print("\nUse the following command to run a simulation:")
print("python run_research.py --scenario custom --config example_config.json")
print("\nModular structure is ready for use!")
