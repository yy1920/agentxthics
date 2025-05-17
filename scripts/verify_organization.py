#!/usr/bin/env python3
"""
Verification script to check that the new directory organization is correct.
"""
import os
import sys
import json

def check_directory(directory, expected_subdirs=None, expected_files=None, min_files=None):
    """Check if a directory exists and contains expected subdirectories and files."""
    if not os.path.exists(directory):
        print(f"❌ ERROR: Directory {directory} does not exist")
        return False
    
    print(f"✅ Directory {directory} exists")
    
    success = True
    
    # Check for expected subdirectories
    if expected_subdirs:
        for subdir in expected_subdirs:
            subdir_path = os.path.join(directory, subdir)
            if not os.path.exists(subdir_path) or not os.path.isdir(subdir_path):
                print(f"❌ ERROR: Expected subdirectory {subdir} not found in {directory}")
                success = False
            else:
                print(f"✅ Found subdirectory {subdir} in {directory}")
    
    # Check for expected files
    if expected_files:
        for file in expected_files:
            file_path = os.path.join(directory, file)
            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                print(f"❌ ERROR: Expected file {file} not found in {directory}")
                success = False
            else:
                print(f"✅ Found file {file} in {directory}")
    
    # Check for minimum number of files
    if min_files:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        if len(files) < min_files:
            print(f"❌ ERROR: Expected at least {min_files} files in {directory}, but found only {len(files)}")
            success = False
        else:
            print(f"✅ Found {len(files)} files in {directory} (expected at least {min_files})")
    
    return success

def check_import(module_name):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✅ Successfully imported {module_name}")
        return True
    except ImportError as e:
        print(f"❌ ERROR: Failed to import {module_name}: {e}")
        return False
    
def check_config_paths():
    """Check if config files have the correct output directories."""
    success = True
    
    # Check main config file
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)
        if config.get("output_dir") != "logs/electricity_game":
            print(f"❌ ERROR: config.json has incorrect output_dir: {config.get('output_dir')}")
            success = False
        else:
            print(f"✅ config.json has correct output_dir: {config.get('output_dir')}")
    else:
        print("❌ ERROR: config.json not found")
        success = False
    
    # Check electricity game config file
    if os.path.exists("config_electricity_game.json"):
        with open("config_electricity_game.json", "r") as f:
            config = json.load(f)
        if config.get("output_dir") != "logs/electricity_game":
            print(f"❌ ERROR: config_electricity_game.json has incorrect output_dir: {config.get('output_dir')}")
            success = False
        else:
            print(f"✅ config_electricity_game.json has correct output_dir: {config.get('output_dir')}")
    else:
        print("❌ ERROR: config_electricity_game.json not found")
        success = False
    
    return success

def main():
    """Check if the new directory organization meets all requirements."""
    print("AgentXthics Directory Organization Verification")
    print("=" * 50)
    
    success = True
    
    # Check main directories
    print("\nVerifying top-level directories:")
    main_dirs = ["agentxthics", "docs", "examples", "logs", "scripts", "tests"]
    for directory in main_dirs:
        if not os.path.exists(directory):
            print(f"❌ ERROR: Main directory {directory} does not exist")
            success = False
        else:
            print(f"✅ Found main directory {directory}")
    
    # Check logs directory structure
    print("\nVerifying logs directory structure:")
    logs_success = check_directory(
        "logs", 
        expected_subdirs=["electricity_game", "docs_data", "research"],
    )
    success = success and logs_success
    
    # Check electricity_game logs directory
    print("\nVerifying electricity_game logs directory:")
    electricity_success = check_directory(
        "logs/electricity_game",
        expected_files=["launch_dashboard.py", "dashboard_static.html"],
        min_files=5
    )
    success = success and electricity_success
    
    # Check scenarios directory
    print("\nVerifying scenarios directory:")
    scenarios_success = check_directory(
        "agentxthics/scenarios",
        expected_files=["electricity_game.py", "modified_electricity_game.py"],
    )
    success = success and scenarios_success
    
    # Check config files
    print("\nVerifying configuration files:")
    config_success = check_config_paths()
    success = success and config_success
    
    # Final verdict
    print("\n" + "=" * 50)
    if success:
        print("✅ All checks passed! The directory organization looks good.")
        print("You can now proceed with development using the new structure.")
    else:
        print("❌ Some checks failed. Please fix the issues listed above.")
    
    print("\nRecommended next step: Run a test simulation to verify everything works")
    print("  python simple_run.py mock config.json")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
