#!/bin/bash
# start.sh -- Quick start script for the Agent

echo "Starting Multi-Layered WhatsApp Agent..."

# Activate the virtual environment
source venv/bin/activate

# 1. Seed knowledge graph from history + open visualization
echo ""
echo "Seeding Knowledge Graph from conversation history..."
python scripts/seed_and_view_graph.py

# 2. Run the main agent
echo ""
echo "Launching Agent..."
python run.py
