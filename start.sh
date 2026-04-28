#!/bin/bash
# =============================================================================
# start.sh -- One-click setup & launcher for the WhatsApp AI Agent
# =============================================================================

set -e

BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"

echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}${BOLD}║       WhatsApp AI Agent — Setup & Launcher       ║${RESET}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════╝${RESET}"
echo ""

# ── Step 1: Python check ──────────────────────────────────────────────────────
echo -e "${BOLD}[1/5] Checking Python version...${RESET}"
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}✗ Python3 not found. Please install Python 3.9+ and re-run.${RESET}"
    exit 1
fi
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}✓ Python $PYTHON_VERSION found.${RESET}"
echo ""

# ── Step 2: Virtual environment ───────────────────────────────────────────────
echo -e "${BOLD}[2/5] Setting up virtual environment...${RESET}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created.${RESET}"
else
    echo -e "${GREEN}✓ Virtual environment already exists.${RESET}"
fi
source venv/bin/activate
echo ""

# ── Step 3: Install dependencies ─────────────────────────────────────────────
echo -e "${BOLD}[3/5] Installing dependencies...${RESET}"
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo -e "${GREEN}✓ Python packages installed.${RESET}"

echo -e "Installing Playwright browsers..."
playwright install chromium --quiet
echo -e "${GREEN}✓ Playwright Chromium installed.${RESET}"
echo ""

# ── Step 4: API Key configuration ────────────────────────────────────────────
echo -e "${BOLD}[4/5] API Key Configuration${RESET}"

if [ -f ".env" ]; then
    echo -e "${YELLOW}⚠  A .env file already exists.${RESET}"
    echo -e "   What would you like to do?"
    echo -e "   ${BOLD}1)${RESET} Keep existing .env (skip this step)"
    echo -e "   ${BOLD}2)${RESET} Update API keys"
    echo -ne "\n   Enter choice [1/2]: "
    read -r ENV_CHOICE
else
    ENV_CHOICE="2"
fi

if [ "$ENV_CHOICE" = "2" ]; then
    echo ""
    echo -e "   You need API keys from the following providers:"
    echo -e "   • ${CYAN}Groq (primary LLM, fast & free tier):${RESET} https://console.groq.com/keys"
    echo -e "   • ${CYAN}Gemini (fallback LLM):${RESET}             https://aistudio.google.com/app/apikey"
    echo ""

    # Groq key
    echo -ne "   ${BOLD}Enter your GROQ_API_KEY${RESET} (or press Enter to skip): "
    read -r GROQ_KEY

    # Gemini key
    echo -ne "   ${BOLD}Enter your GEMINI_API_KEY${RESET} (or press Enter to skip): "
    read -r GEMINI_KEY

    # Write .env
    {
        echo "GROQ_API_KEY=${GROQ_KEY}"
        echo "GEMINI_API_KEY=${GEMINI_KEY}"
    } > .env

    echo ""
    echo -e "${GREEN}✓ .env file saved.${RESET}"

    # Warn if both are empty
    if [ -z "$GROQ_KEY" ] && [ -z "$GEMINI_KEY" ]; then
        echo -e "${RED}⚠  No API keys provided. The agent will not be able to generate responses.${RESET}"
        echo -e "   Edit ${BOLD}.env${RESET} manually before running."
    elif [ -z "$GROQ_KEY" ]; then
        echo -e "${YELLOW}⚠  GROQ_API_KEY not set — will use Gemini only (no fallback chain).${RESET}"
    elif [ -z "$GEMINI_KEY" ]; then
        echo -e "${YELLOW}⚠  GEMINI_API_KEY not set — Groq is primary; no fallback if Groq fails.${RESET}"
    fi
else
    echo -e "${GREEN}✓ Keeping existing .env.${RESET}"
fi
echo ""

# ── Step 5: Launch ────────────────────────────────────────────────────────────
echo -e "${BOLD}[5/5] Launching the Agent${RESET}"
echo ""
echo -e "   What would you like to do?"
echo -e "   ${BOLD}1)${RESET} Run the agent (requires a contact name)"
echo -e "   ${BOLD}2)${RESET} Seed & view Knowledge Graph"
echo -e "   ${BOLD}3)${RESET} Both (seed graph, then run agent)"
echo -e "   ${BOLD}4)${RESET} Exit"
echo -ne "\n   Enter choice [1/2/3/4]: "
read -r RUN_CHOICE

case "$RUN_CHOICE" in
    1|3)
        echo -ne "\n   ${BOLD}Run in headless mode?${RESET} (browser runs invisibly in background)"
        echo -ne "\n   ${BOLD}[Y/n]:${RESET} "
        read -r HEADLESS_CHOICE

        HEADFUL_FLAG=""
        if [[ "$HEADLESS_CHOICE" =~ ^[Nn]$ ]]; then
            HEADFUL_FLAG="--headful"
            echo -e "   ${YELLOW}→ Browser window will be visible.${RESET}"
        else
            echo -e "   ${GREEN}→ Running headless (no browser window).${RESET}"
        fi

        if [ "$RUN_CHOICE" = "3" ]; then
            echo ""
            echo -e "${GREEN}Seeding Knowledge Graph...${RESET}"
            python scripts/seed_and_view_graph.py
        fi

        echo -ne "\n   ${BOLD}Enter the WhatsApp contact name to monitor:${RESET} "
        read -r CONTACT
        echo ""
        echo -e "${GREEN}Starting agent for \"$CONTACT\"...${RESET}"
        python run.py --contact "$CONTACT" $HEADFUL_FLAG
        ;;
    2)
        echo ""
        echo -e "${GREEN}Seeding Knowledge Graph...${RESET}"
        python scripts/seed_and_view_graph.py
        ;;
    4)
        echo -e "${YELLOW}Exiting. Run ${BOLD}./start.sh${RESET}${YELLOW} again when ready.${RESET}"
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${RESET}"
        exit 1
        ;;
esac

