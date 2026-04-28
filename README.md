# Multi-Layered Autonomous WhatsApp Agent

A cognitive architecture for autonomous WhatsApp conversation management, powered by advanced LLMs like Groq and Gemini.

## Features

- **Multi-Layered Architecture**: Orchestrator, Perception, Strategist, Tactician, Executor, and Reflection layers.
- **LLM Integration**: Primary support for Groq (fast inference) with a fallback to Gemini.
- **Knowledge Graph**: Automatic extraction of relational facts visualized with vis.js.
- **Browser Automation**: Playwright-based WhatsApp Web interaction.
- **Human Mimicry**: Natural typing delays, silence policies, and context analysis.

## Setup Instructions

### 1. Install Dependencies

Ensure you have Python 3.9+ installed.

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

### 2. Environment Variables

Create a `.env` file in the root directory and add your API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

*Note: Groq is the primary model used for fast reasoning. Gemini is used as a fallback if Groq is unavailable or rate-limited.*

### 3. Google Calendar Setup (Optional)

1. Enable Google Calendar API in [Google Cloud Console](https://console.cloud.google.com).
2. Create OAuth2 credentials (Desktop app).
3. Download `credentials.json` to the project root.
4. On first run, a browser window opens for OAuth consent.
5. Token is cached automatically.

*Without credentials, the agent uses a local JSON calendar.*

## Running the Agent

Start the agent and specify the contact you want to monitor/interact with:

```bash
python run.py --contact "Target Contact Name"
```

With debug logging to see the cognitive layers in action:

```bash
python run.py --contact "Target Contact Name" --debug-layers --verbose
```

## Knowledge Graph Visualization

To generate and view the knowledge graph from conversation history:

```bash
python scripts/seed_and_view_graph.py
```

## Cognitive Cycle

Each loop iteration runs 13 phases:

```
Perceive → Classify → Bootstrap → SilencePolicy → BusinessGate
→ GoalEval → Strategize → ContextAnalyze → Plan
→ MessageGuard → Execute → Reflect → Maintain
```
