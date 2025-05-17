#!/bin/bash

# -------------------------------------------------------------------------
# Clean Project Script
# This script helps identify and remove redundant files now that the 
# project has been reorganized into a clean structure.
# -------------------------------------------------------------------------

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}AgentXthics Project Cleanup Utility${NC}"
echo "--------------------------------------"
echo "This tool identifies redundant files and directories that can be safely removed."
echo 

if [ ! -d "agentxthics" ]; then
    echo -e "${RED}Error: Main package directory not found at ./agentxthics${NC}"
    exit 1
fi

if [ ! -d "results" ]; then
    echo -e "${RED}Error: Results directory not found at ./results${NC}"
    exit 1
fi

# Files/directories to exclude from cleanup
EXCLUDE_PATTERNS=(
    "*.egg-info"
    "__pycache__"
    "*.pyc"
    "*.xlsx"
    "*.csv"
    "*.log"
    "*.json"
    "decision_analysis.png"
    "message_analysis.png"
    "Electricity Game"
    "Admin.xlsx"
    "Meeting Notes*"
    "full_log.txt"
    "public_log.txt"
    "electricity_game_results"
    "docs/research"
)

# Function to check if a file should be excluded
should_exclude() {
    local file="$1"
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
        if [[ "$file" == *"$pattern"* ]]; then
            return 0 # exclude
        fi
    done
    return 1 # include
}

echo -e "${YELLOW}Scanning for redundant files...${NC}"

# Create temporary file lists
find electricity_game_results -type f | grep -v "__pycache__" | sort > /tmp/original_files.txt
find results -type f | grep -v "__pycache__" | sort > /tmp/organized_files.txt

# Filter unnecessary files from the lists
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    grep -v "$pattern" /tmp/original_files.txt > /tmp/original_filtered.txt
    mv /tmp/original_filtered.txt /tmp/original_files.txt
done

echo
echo -e "${GREEN}Redundant Files Analysis:${NC}"
echo "--------------------------------------"
echo "The following files from the original structure have been"
echo "reorganized and can be safely removed:"
echo

# Original files that are now organized
count=0
while read -r file; do
    base_name=$(basename "$file")
    if grep -q "$base_name" /tmp/organized_files.txt && ! should_exclude "$file"; then
        echo "- $file"
        count=$((count+1))
    fi
done < /tmp/original_files.txt

if [ $count -eq 0 ]; then
    echo "No redundant files found."
else
    echo
    echo -e "${YELLOW}Found $count redundant files that can be safely removed.${NC}"
fi

echo
echo -e "${GREEN}What would you like to do?${NC}"
echo "1. Remove all redundant files (permanently deletes files)"
echo "2. Create a cleanup plan only (no files will be deleted)"
echo "3. Exit without changes"
echo
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo
        echo -e "${YELLOW}Removing redundant files...${NC}"
        echo
        
        count=0
        while read -r file; do
            base_name=$(basename "$file")
            if grep -q "$base_name" /tmp/organized_files.txt && ! should_exclude "$file"; then
                echo "Removing: $file"
                rm "$file"
                count=$((count+1))
            fi
        done < /tmp/original_files.txt
        
        echo
        echo -e "${GREEN}Removed $count redundant files.${NC}"
        ;;
    2)
        echo
        echo -e "${YELLOW}Creating cleanup plan...${NC}"
        echo
        
        echo "# AgentXthics Cleanup Plan" > cleanup_plan.md
        echo "The following files can be safely removed from the original structure:" >> cleanup_plan.md
        echo >> cleanup_plan.md
        
        count=0
        while read -r file; do
            base_name=$(basename "$file")
            if grep -q "$base_name" /tmp/organized_files.txt && ! should_exclude "$file"; then
                echo "- \`$file\`" >> cleanup_plan.md
                count=$((count+1))
            fi
        done < /tmp/original_files.txt
        
        echo >> cleanup_plan.md
        echo "Total redundant files: $count" >> cleanup_plan.md
        
        echo -e "${GREEN}Cleanup plan created: cleanup_plan.md${NC}"
        ;;
    3)
        echo
        echo -e "${YELLOW}Exiting without changes.${NC}"
        ;;
    *)
        echo
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        ;;
esac

# Clean up temporary files
rm -f /tmp/original_files.txt /tmp/organized_files.txt

echo
echo -e "${GREEN}Cleanup process complete.${NC}"
