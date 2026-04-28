"""
scripts/seed_and_view_graph.py -- Knowledge Graph seeder + vis.js visualizer.

Extracts relational facts from all conversations and renders them as a
beautiful, interactive HTML graph using vis.js.

Features:
  - Contact nodes (people) are visually distinct from concept nodes
  - Hovering a contact node shows their recent messages
  - Physics-based layout spreads nodes naturally
"""

import os, sys, json, re, asyncio, textwrap, subprocess, html as html_escape_mod

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.database import ConversationDB
from core.llm_client import LLMClient

SKIP_VALUES = {"subject", "predicate", "object", "entity", "name", "relationship",
               "person", "place", "organization", "thing"}

EXTRACTION_PROMPT = """\
You are a Knowledge Graph extraction engine.
Read the conversation and extract every meaningful relational fact.

Return ONLY a single valid JSON array. Each element is a 3-element string array:
  ["Subject", "relationship", "Object"]

Example output:
[["Alice", "studies at", "University"], ["Bob", "lives in", "New York"], ["Charlie", "works at", "Tech Corp"]]

Rules:
- Output ONLY valid JSON. No markdown, no explanation, no extra text.
- Each triple must be ["string","string","string"] with REAL names/places/roles.
- If nothing meaningful exists, return exactly: []
"""

# Contact nodes — large, distinct "star" shape
CONTACT_COLOR  = "#00ffcc"
CONTACT_SIZE   = 36
CONTACT_SHAPE  = "star"

# Concept nodes — small dots in varied palette colours
CONCEPT_PALETTE = [
    "#ff6b9d","#ffd93d","#6bcbff",
    "#c3ff6b","#ff9f6b","#d0b4ff","#ff6b6b",
    "#6bffb4","#ffb46b","#6b9dff","#ff6bff",
]
CONCEPT_SIZE   = 18
CONCEPT_SHAPE  = "dot"


def parse_triples(raw: str) -> list:
    triples = []
    for m in re.finditer(r'\[(?:[^\[\]]|\[(?:[^\[\]])*\])*\]', raw, re.DOTALL):
        try:
            data = json.loads(m.group(0))
            if not data:
                continue
            if isinstance(data[0], list):
                triples.extend(data)
            elif isinstance(data[0], str) and len(data) == 3:
                triples.append(data)
        except Exception:
            continue
    return triples


async def extract(llm, contact_name: str, messages: list) -> list:
    if not messages:
        return []
    convo = "\n".join(f"[{m['role'].upper()}]: {m['content']}" for m in messages[-60:])
    prompt = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user",   "content": f"Contact: {contact_name}\n\n{convo}"},
    ]
    try:
        raw = await llm.chat(prompt)
        clean = []
        for t in parse_triples(raw):
            if len(t) != 3:
                continue
            s, p, o = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()
            if s.lower() in SKIP_VALUES or o.lower() in SKIP_VALUES or not s or not o:
                continue
            clean.append([s, p, o])
        return clean
    except Exception as e:
        print(f"  ⚠  LLM error for {contact_name}: {e}")
        return []


def build_tooltip(contact_name: str, messages: list) -> str:
    """Build a rich HTML tooltip showing the last 6 messages."""
    if not messages:
        return f"<b>{html_escape_mod.escape(contact_name)}</b><br/><i>No messages yet</i>"

    rows = ""
    for m in messages[-6:]:
        role = m["role"]
        icon = "🧑" if role == "user" else "🤖"
        label_color = "#94a3b8" if role == "user" else "#00ffcc"
        content = html_escape_mod.escape(m["content"][:80])
        if len(m["content"]) > 80:
            content += "…"
        rows += (
            f'<div style="margin-bottom:6px;">'
            f'<span style="color:{label_color};font-size:11px;">{icon} {role.upper()}</span><br/>'
            f'<span style="color:#e2e8f0;font-size:13px;">{content}</span>'
            f'</div>'
        )

    return (
        f'<div style="max-width:280px;font-family:system-ui;">'
        f'<div style="font-size:15px;font-weight:700;color:#00ffcc;margin-bottom:8px;'
        f'border-bottom:1px solid #334155;padding-bottom:6px;">'
        f'💬 {html_escape_mod.escape(contact_name)}</div>'
        f'{rows}'
        f'</div>'
    )


def build_html(triples: list, contact_data: dict) -> str:
    """
    Generate a self-contained vis.js graph HTML file.

    contact_data: {name: [messages_list]} for all WhatsApp contacts.
    """
    node_ids: dict[str, int] = {}
    node_colors: dict[str, str] = {}
    edges_data: list[dict] = []
    nid = 0
    concept_color_idx = 0
    contact_names = set(contact_data.keys())

    def get_id(label: str) -> int:
        nonlocal nid, concept_color_idx
        if label not in node_ids:
            node_ids[label] = nid
            if label in contact_names:
                node_colors[label] = CONTACT_COLOR
            else:
                node_colors[label] = CONCEPT_PALETTE[concept_color_idx % len(CONCEPT_PALETTE)]
                concept_color_idx += 1
            nid += 1
        return node_ids[label]

    for s, p, o in triples:
        sid, oid = get_id(s), get_id(o)
        edges_data.append({"from": sid, "to": oid, "label": p})

    nodes_json = json.dumps([
        {
            "id": nid_val,
            "label": label,
            "title": (
                build_tooltip(label, contact_data[label])
                if label in contact_names
                else f"<b>{html_escape_mod.escape(label)}</b>"
            ),
            "color": {
                "background": node_colors[label],
                "border": "#ffffff55" if label in contact_names else "#ffffff22",
                "highlight": {"background": "#ffffff", "border": "#ffffff"},
            },
            "font": {
                "color": "#ffffff",
                "size": 17 if label in contact_names else 13,
                "bold": True,
                "strokeWidth": 3,
                "strokeColor": "#00000099",
            },
            "shape": CONTACT_SHAPE if label in contact_names else CONCEPT_SHAPE,
            "size": CONTACT_SIZE if label in contact_names else CONCEPT_SIZE,
            "group": "contact" if label in contact_names else "concept",
        }
        for label, nid_val in node_ids.items()
    ], indent=2)

    edges_json = json.dumps([
        {
            "from": e["from"], "to": e["to"],
            "label": e["label"],
            "arrows": "to",
            "color": {"color": "#6b728088", "highlight": "#ffffff"},
            "font": {"color": "#9ca3af", "size": 11, "align": "middle"},
            "smooth": {"type": "curvedCW", "roundness": 0.15},
        }
        for e in edges_data
    ], indent=2)

    n_contacts = sum(1 for l in node_ids if l in contact_names)
    n_concepts = len(node_ids) - n_contacts
    stats = (
        f"<span style='color:#00ffcc'>★ {n_contacts} contacts</span>"
        f"&nbsp;·&nbsp;"
        f"<span style='color:#9ca3af'>● {n_concepts} concepts</span>"
        f"&nbsp;·&nbsp;"
        f"<span style='color:#64748b'>{len(edges_data)} relationships</span>"
    )

    legend_html = (
        "<b>Legend</b><br/>"
        "<span style='color:#00ffcc'>★</span> Contact (hover for messages)<br/>"
        "<span style='color:#6bcbff'>●</span> Concept / Place / Org<br/>"
        "<hr style='border-color:#334155;margin:6px 0'/>"
        "<b>Controls</b><br/>"
        "Scroll to zoom<br/>"
        "Drag to pan<br/>"
        "Click node to highlight<br/>"
        "Hover for details"
    )

    return textwrap.dedent(f"""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
      <title>Aria · Knowledge Graph</title>
      <script src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
      <style>
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ background: #0f172a; font-family: 'Inter', system-ui, sans-serif; color: #e2e8f0; height: 100vh; overflow: hidden; }}
        #header {{
          position: fixed; top: 0; left: 0; right: 0; z-index: 10;
          background: linear-gradient(90deg,#1e293b,#0f172a);
          border-bottom: 1px solid #334155;
          padding: 12px 24px; display: flex; align-items: center; gap: 20px;
        }}
        #header h1 {{ font-size: 17px; font-weight: 700; letter-spacing: -0.3px; color: #00ffcc; white-space:nowrap; }}
        #header .stats {{ font-size: 13px; display:flex; gap: 12px; flex-wrap:wrap; }}
        #mynetwork {{ width: 100%; height: calc(100vh - 50px); margin-top: 50px; }}
        #legend {{
          position: fixed; bottom: 20px; right: 20px; z-index: 10;
          background: #1e293bdd; border: 1px solid #334155; border-radius: 12px;
          padding: 12px 16px; font-size: 12px; color: #94a3b8; line-height: 1.7;
          backdrop-filter: blur(8px);
        }}
        /* Style the vis.js tooltip */
        div.vis-tooltip {{
          background: #1e293b !important;
          border: 1px solid #334155 !important;
          border-radius: 10px !important;
          padding: 10px 14px !important;
          box-shadow: 0 8px 32px #00000088 !important;
          max-width: 300px !important;
          font-size: 13px !important;
          color: #e2e8f0 !important;
          pointer-events: none !important;
        }}
      </style>
    </head>
    <body>
      <div id="header">
        <h1>⚡ Aria · Knowledge Graph</h1>
        <div class="stats">{stats}</div>
      </div>
      <div id="mynetwork"></div>
      <div id="legend">{legend_html}</div>
      <script>
        const nodes = new vis.DataSet({nodes_json});
        const edges = new vis.DataSet({edges_json});
        const container = document.getElementById('mynetwork');
        const options = {{
          physics: {{
            solver: 'forceAtlas2Based',
            forceAtlas2Based: {{
              gravitationalConstant: -80,
              centralGravity: 0.002,
              springLength: 200,
              springConstant: 0.06,
              damping: 0.5,
              avoidOverlap: 1.0,
            }},
            stabilization: {{ iterations: 400, fit: true }},
          }},
          interaction: {{
            hover: true,
            tooltipDelay: 100,
            zoomView: true,
            navigationButtons: false,
          }},
          edges: {{ width: 1.5, selectionWidth: 3 }},
          nodes: {{ borderWidth: 2 }},
        }};
        const network = new vis.Network(container, {{ nodes, edges }}, options);
        network.once('stabilizationIterationsDone', () => {{ network.fit({{ animation: true }}); }});
      </script>
    </body>
    </html>
    """)


async def main():
    print("\n╔══════════════════════════════════════╗")
    print("║   Knowledge Graph Seeder & Viewer   ║")
    print("╚══════════════════════════════════════╝\n")

    db = ConversationDB()
    llm = LLMClient()

    db._conn.execute("DELETE FROM knowledge_graph")
    db._conn.commit()
    print("Cleared previous graph data.\n")

    contacts = db._conn.execute("SELECT id, name FROM contacts ORDER BY id").fetchall()
    if not contacts:
        print("No contacts found. Run the agent first.")
        db.close(); return

    print(f"Found {len(contacts)} contact(s). Extracting knowledge...\n")
    total = 0
    # contact_name → recent messages (for tooltip)
    contact_data: dict = {}

    for contact_id, contact_name in contacts:
        messages = db.get_all_messages_for_summary(contact_id, limit=100)
        recent = db.get_recent_messages(contact_id, limit=6)
        contact_data[contact_name] = recent  # always register even if 0 messages

        if not messages:
            print(f"  • {contact_name}: no messages, skipping.")
            continue

        print(f"  • {contact_name} ({len(messages)} msgs) — extracting...")
        triples = await extract(llm, contact_name, messages)
        db.add_knowledge_triple(contact_name, "contact of", "User")
        for t in triples:
            db.add_knowledge_triple(t[0], t[1], t[2])
            total += 1
        print(f"    ✓ {len(triples)} facts extracted.")

    print(f"\nTotal facts extracted: {total}")

    all_triples = db.get_full_knowledge_graph()
    db.close()

    if not all_triples:
        print("\nNot enough data yet. Keep chatting and re-run later!"); return

    raw_triples = [[t["subject"], t["predicate"], t["object"]] for t in all_triples]
    html = build_html(raw_triples, contact_data)

    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "knowledge_map.html"
    )
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nRendered {len(all_triples)} relationships → {out}")
    print("Opening Knowledge Graph in browser...\n")
    subprocess.Popen(["open", out])


if __name__ == "__main__":
    asyncio.run(main())
