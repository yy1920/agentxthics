#!/usr/bin/env python
"""
Diagnostic script for AgentXthics installation.
This script helps identify common installation and import issues.
"""
import os
import sys
import importlib
import traceback

# Set up color output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def print_success(message):
    print(f"{GREEN}✓ {message}{RESET}")

def print_warning(message):
    print(f"{YELLOW}⚠ {message}{RESET}")

def print_error(message):
    print(f"{RED}✗ {message}{RESET}")

def print_header(message):
    print(f"\n{YELLOW}=== {message} ==={RESET}")

def check_path():
    print_header("Python Path")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print("\nPython path:")
    for path in sys.path:
        print(f"  {path}")
    
    # Check if current directory is in path
    cwd = os.getcwd()
    if cwd in sys.path:
        print_success(f"Current directory ({cwd}) is in Python path")
    else:
        print_warning(f"Current directory ({cwd}) is NOT in Python path")
        
    # Check parent directory
    parent_dir = os.path.dirname(cwd)
    if parent_dir in sys.path:
        print_success(f"Parent directory ({parent_dir}) is in Python path")
    else:
        print_warning(f"Parent directory ({parent_dir}) is NOT in Python path")

def check_imports():
    print_header("Import Tests")
    modules_to_check = [
        "agentxthics",
        "agentxthics.research",
        "agentxthics.research.scenarios",
        "agentxthics.research.analysis",
        "agentxthics.research.metrics",
        "agentxthics.agents",
        "agentxthics.frameworks",
        "agentxthics.resources"
    ]
    
    for module_name in modules_to_check:
        try:
            module = importlib.import_module(module_name)
            print_success(f"Successfully imported {module_name}")
        except ImportError as e:
            print_error(f"Failed to import {module_name}: {e}")
    
    # Try direct import
    print("\nTrying direct imports:")
    try:
        import research.scenarios
        print_success("Successfully imported research.scenarios")
    except ImportError as e:
        print_error(f"Failed to import research.scenarios: {e}")
        
    try:
        from research.scenarios import run_simulation
        print_success("Successfully imported run_simulation from research.scenarios")
    except ImportError as e:
        print_error(f"Failed to import run_simulation: {e}")

def check_file_structure():
    print_header("File Structure")
    directories_to_check = [
        "agentxthics",
        "agentxthics/research",
        "agentxthics/agents",
        "agentxthics/frameworks",
        "agentxthics/resources"
    ]
    
    cwd = os.getcwd()
    for directory in directories_to_check:
        path = os.path.join(cwd, directory)
        if os.path.exists(path) and os.path.isdir(path):
            print_success(f"Directory exists: {directory}")
            # Check for __init__.py
            init_path = os.path.join(path, "__init__.py")
            if os.path.exists(init_path):
                print_success(f"  __init__.py exists in {directory}")
            else:
                print_warning(f"  __init__.py missing from {directory}")
        else:
            print_error(f"Directory missing: {directory}")

def check_installation():
    print_header("Installation Check")
    try:
        import pkg_resources
        distributions = [d for d in pkg_resources.working_set]
        installed = any(d.project_name == "agentxthics" for d in distributions)
        
        if installed:
            print_success("agentxthics is installed in development mode")
            
            # Get the package path
            package = pkg_resources.get_distribution("agentxthics")
            print(f"Installed at: {package.location}")
        else:
            print_warning("agentxthics is not installed")
            print("Try running: pip install -e .")
    except Exception as e:
        print_error(f"Error checking installation: {e}")

def provide_recommendations():
    print_header("Recommendations")
    print(f"{YELLOW}If you are experiencing import errors, try:{RESET}")
    print("1. Install the package in development mode:")
    print("   pip install -e .")
    print("\n2. Make sure all directories have __init__.py files")
    print("\n3. Use relative imports within the package:")
    print("   from ..research import scenarios  # When importing from a sibling module")
    print("   from . import base  # When importing from the same directory")
    print("\n4. When running scripts directly, use the -m flag:")
    print("   python -m agentxthics.run_research")

if __name__ == "__main__":
    check_path()
    check_file_structure()
    check_imports()
    check_installation()
    provide_recommendations()
