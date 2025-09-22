#!/bin/bash
# Example script to run a complete forensic analysis

# Set up environment
export CASE_ID="CASE_$(date +%Y%m%d_%H%M%S)"
export ANALYST="demo_analyst"

echo "Starting Smart Forensic Bot Analysis"
echo "Case ID: $CASE_ID"
echo "Analyst: $ANALYST"

# Initialize the system
python3 main.py init

# Import knowledge base
python3 main.py import-knowledge examples/knowledge_base.json

# Create some sample evidence files for demonstration
mkdir -p evidence
echo "This is a sample log file with suspicious activity" > evidence/sample.log
echo "Suspicious script content" > evidence/suspicious_script.bat
echo "Binary content simulation" > evidence/sample.exe

# Run analysis
python3 main.py analyze evidence/ --case-id "$CASE_ID" --analyst "$ANALYST" --output "reports/analysis_${CASE_ID}.json" --recursive

# Show knowledge graph stats
python3 main.py graph-stats

echo "Analysis complete! Check the reports directory for results."