# WhatsApp AI Agent

An autonomous, multi-layered AI agent that manages WhatsApp conversations with human-like intelligence — powered by **Groq** (primary) and **Gemini** (fallback).

---

## Quick Start

```bash
git clone https://github.com/priteshraj10/whatsapp-AIagent.git
cd whatsapp-AIagent
chmod +x start.sh
./start.sh
```

That's it. `start.sh` handles **everything automatically**:
- Creates a Python virtual environment
- Installs all dependencies from `requirements.txt`
- Installs Playwright + Chromium browser
- Prompts you to enter your API keys and saves them to `.env`
- Lets you choose what to launch

---

## API Keys Required

| Provider | Purpose | Get Key |
|---|---|---|
| **Groq** | Primary LLM (fast, free tier) | https://console.groq.com/keys |
| **Gemini** | Fallback LLM | https://aistudio.google.com/app/apikey |

When you run `./start.sh`, it will ask you to enter these keys interactively. They are saved locally to a `.env` file (never committed to git).

> You can also create `.env` manually:
> ```env
> GROQ_API_KEY=your_groq_api_key_here
> GEMINI_API_KEY=your_gemini_api_key_here
> ```

---

## How It Works

The agent runs a **13-phase cognitive cycle** every loop iteration:

```
Perceive -> Classify -> Bootstrap -> SilencePolicy -> BusinessGate
-> GoalEval -> Strategize -> ContextAnalyze -> Plan
-> MessageGuard -> Execute -> Reflect -> Maintain
```

Each phase is handled by a dedicated cognitive layer:

| Layer | File | Role |
|---|---|---|
| Orchestrator | `agents/orchestrator.py` | Central coordinator, runs the cycle |
| Perception | `agents/perception.py` | Signal classification (no LLM cost) |
| Strategist | `agents/strategist.py` | High-level intent planning |
| Tactician | `agents/tactician.py` | Tool-augmented reasoning |
| Executor | `agents/executor.py` | Action dispatch + human mimicry |
| Reflection | `agents/reflection.py` | Metacognitive self-evaluation |
| Goal Manager | `agents/goal_manager.py` | Persistent goal stack |
| Knowledge | `agents/knowledge.py` | Relational fact extraction |

---

## Project Structure

```
whatsapp-AIagent/
├── agents/              # Cognitive layers
├── core/                # LLM client, tools, profiler
├── engine/              # Playwright browser lifecycle
├── integrations/        # Database & Calendar APIs
├── skills/              # Pre-action intelligence modules
├── whatsapp/            # DOM reader & CSS selectors
├── scripts/             # Utilities (knowledge graph, contact discovery)
├── persona/             # Agent personality config (YAML + skills)
├── start.sh             # One-click setup & launcher  <-- START HERE
├── run.py               # CLI entry point
└── requirements.txt     # Python dependencies
```

---

## Manual Usage

If you prefer to run things manually:

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
playwright install chromium

# 3. Create .env with your API keys
cp .env.example .env            # Then edit .env

# 4. Run the agent
python run.py --contact "Contact Name"

# With verbose debug logging
python run.py --contact "Contact Name" --debug-layers --verbose
```

---

## Knowledge Graph

Visualize what the agent has learned about your contacts:

```bash
python scripts/seed_and_view_graph.py
```

This generates an interactive vis.js graph and opens it in your browser.

---

## Google Calendar Integration (Optional)

1. Enable Google Calendar API in [Google Cloud Console](https://console.cloud.google.com)
2. Create OAuth2 credentials (Desktop App)
3. Download `credentials.json` to the project root
4. First run opens a browser for OAuth consent — token is cached automatically

Without credentials, the agent uses a local JSON calendar file.

---

## Requirements

- Python 3.9+
- A Groq API key (free tier available)
- macOS / Linux (Windows supported via WSL)
