"""
scripts/visualize_graph.py -- Exports the agent's Knowledge Graph to an interactive HTML map.

Run this script to generate `knowledge_map.html` in the root directory.
"""

import os
import sys
import webbrowser
from pyvis.network import Network

# Ensure we can import the database from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from integrations.database import ConversationDB

def generate_graph():
    print("Connecting to database...")
    db = ConversationDB()
    
    triples = db.get_full_knowledge_graph()
    db.close()
    
    if not triples:
        print("Knowledge Graph is currently empty! The agent needs to extract some facts first.")
        return

    print(f"Found {len(triples)} relationships. Building graph...")
    
    # Initialize a directed PyVis network
    net = Network(
        height="800px", 
        width="100%", 
        bgcolor="#1e1e1e", 
        font_color="white",
        directed=True
    )
    
    # Optional: use a nice physics layout
    net.force_atlas_2based()
    
    # Add nodes and edges
    for t in triples:
        subj = t["subject"]
        pred = t["predicate"]
        obj = t["object"]
        
        # Add nodes (PyVis handles duplicates automatically if we pass the same ID)
        net.add_node(subj, label=subj, title=subj, color="#00ffcc")
        net.add_node(obj, label=obj, title=obj, color="#ff00cc")
        
        # Add edge
        net.add_edge(subj, obj, title=pred, label=pred, color="#666666")

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_map.html")
    
    print(f"Saving visualization to {output_path}...")
    net.save_graph(output_path)
    
    print("Done! Opening in browser...")
    webbrowser.open(f"file://{output_path}")

if __name__ == "__main__":
    generate_graph()
