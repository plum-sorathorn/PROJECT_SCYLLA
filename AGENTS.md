# AGENTS.md — AI Assistant Configuration for PROJECT SCYLLA

This document describes the AI assistant integrations and agent workflows configured for this project.

## Overview

PROJECT SCYLLA uses multiple AI agent configurations to assist with development, code navigation, and documentation workflows. These configurations are stored in dedicated directories and enable intelligent code assistance.

---

## Agent Directories

### `.agents/`
Primary agent configuration directory containing rules and workflows.

```
.agents/
├── rules/       # Behavioral rules applied to AI agents
│   └── graphify.md
└── workflows/   # Reusable agent workflows
    └── graphify.md
```

### `.gemini/`
Google Gemini CLI configuration directory.

```
.gemini/
└── settings.json   # Tool hooks and agent behavior settings
```

---

## Configured Agents & Workflows

### graphify — Knowledge Graph Navigator

**Purpose:** Transforms the codebase into a navigable knowledge graph for efficient code exploration and architecture understanding.

**Trigger:** Always-on (automatically active for codebase questions)

**Rules (`graphify.md`):**
- Consult `graphify-out/` knowledge graph before using grep/directory listing
- Use `graphify query "<question>"` for focused, scoped subgraph results
- Use `graphify path "<A>" "<B>"` to trace relationships between components
- Use `graphify explain "<concept>"` for focused module summaries
- After code modifications, run `graphify update .` to sync the graph

**CLI Commands:**
```powershell
# Query the knowledge graph
graphify query "unusual options scanner"

# Find path between components
graphify path "main.py" "CrowServer"

# Explain a concept/module
graphify explain "IV Skew"

# Update graph after code changes
graphify update .
```

**Output Location:** `graphify-out/`
- `graph.json` — Structured graph data
- `GRAPH_REPORT.md` — Full architecture report
- `wiki/index.md` — Navigation wiki (if generated)

---

## Gemini CLI Integration

**Settings File:** `.gemini/settings.json`

Configures tool hooks for the Gemini CLI assistant:
- **BeforeTool Hook:** Intercepts `read_file` and `list_directory` calls to provide context about the graphify knowledge graph when available
- Guides the agent to use graphify query instead of raw file exploration for better context

---

## Workflow Guidelines

### When to Use Each Agent

| Task | Recommended Approach |
|------|---------------------|
| Locate specific code | `graphify query` first, then read files |
| Understand architecture | `graphify explain` or read `GRAPH_REPORT.md` |
| Trace component relationships | `graphify path` |
| General file exploration | Only if graphify doesn't have the answer |
| After modifying code | `graphify update .` |

### Best Practices
# AGENTS.md — AI Assistant Configuration for PROJECT SCYLLA

This document describes the AI assistant integrations and agent workflows configured for this project.

## Overview

PROJECT SCYLLA uses multiple AI agent configurations to assist with development, code navigation, and documentation workflows. These configurations are stored in dedicated directories and enable intelligent code assistance.

---

## Agent Directories

### `.agents/`
Primary agent configuration directory containing rules and workflows.

```
.agents/
├── rules/       # Behavioral rules applied to AI agents
│   └── graphify.md
└── workflows/   # Reusable agent workflows
    └── graphify.md
```

### `.gemini/`
Google Gemini CLI configuration directory.

```
.gemini/
└── settings.json   # Tool hooks and agent behavior settings
```

---

## Configured Agents & Workflows

### graphify — Knowledge Graph Navigator

**Purpose:** Transforms the codebase into a navigable knowledge graph for efficient code exploration and architecture understanding.

**Trigger:** Always-on (automatically active for codebase questions)

**Rules (`graphify.md`):**
- Consult `graphify-out/` knowledge graph before using grep/directory listing
- Use `graphify query "<question>"` for focused, scoped subgraph results
- Use `graphify path "<A>" "<B>"` to trace relationships between components
- Use `graphify explain "<concept>"` for focused module summaries
- After code modifications, run `graphify update .` to sync the graph

**CLI Commands:**
```powershell
# Query the knowledge graph
graphify query "unusual options scanner"

# Find path between components
graphify path "main.py" "CrowServer"

# Explain a concept/module
graphify explain "IV Skew"

# Update graph after code changes
graphify update .
```

**Output Location:** `graphify-out/`
- `graph.json` — Structured graph data
- `GRAPH_REPORT.md` — Full architecture report
- `wiki/index.md` — Navigation wiki (if generated)

---

## Gemini CLI Integration

**Settings File:** `.gemini/settings.json`

Configures tool hooks for the Gemini CLI assistant:
- **BeforeTool Hook:** Intercepts `read_file` and `list_directory` calls to provide context about the graphify knowledge graph when available
- Guides the agent to use graphify query instead of raw file exploration for better context

---

## Workflow Guidelines

### When to Use Each Agent

| Task | Recommended Approach |
|------|---------------------|
| Locate specific code | `graphify query` first, then read files |
| Understand architecture | `graphify explain` or read `GRAPH_REPORT.md` |
| Trace component relationships | `graphify path` |
| General file exploration | Only if graphify doesn't have the answer |
| After modifying code | `graphify update .` |

### Best Practices

1. **Graph-First Navigation:** Always query the knowledge graph before grepping or listing directories
2. **Scoped Exploration:** Use query/path/explain for focused results — cheaper than manual exploration
3. **Keep Graph Current:** Run `graphify update .` after code changes
4. **Direct File Reads OK:** Once graphify points to specific files, read them directly

---

## System Architecture & Optimization Conventions

### ML Engine & Walk-Forward Parallelization

- **Parallel Step Execution**: Walk-forward backtests are parallelized across CPU cores using `joblib.Parallel` in `backend/routers/ml_model.py`.
- **Thread Lock Concurrency Guard**: A module-level `_walkforward_lock = threading.Lock()` guarantees that only one walk-forward retraining loop executes at a time. Queued requests automatically re-check and reuse disk caches (`cache_predictions_walkforward.pkl`) upon acquiring the lock.
- **Walk-Forward Defaults**: `walkforward_train_window = 500` trades, `walkforward_test_increment = 100` trades.
- **Lookback Period Default**: `lookback_days = null` / `0` (`ALL DATA`).

### Server & Network Ports

- **Python ODP Backend**: Runs on `http://127.0.0.1:6900` via Uvicorn (`python -m uvicorn main:app --host 127.0.0.1 --port 6900`).
- **C++ Core Engine (Optional)**: Listens on `http://127.0.0.1:8080`.
- **Single-Click Launcher**: Execute `LAUNCH_SCYLLA.bat` or `LAUNCH_SCYLLA.ps1` to automatically clear conflicting ports, start the backend server, and launch the web interface.

### Knowledge Graph Maintenance

- After modifying any backend or frontend code files, always execute `graphify update .` to keep the codebase knowledge graph synchronized.

---

## Adding New Agents

To add a new agent workflow:

1. Create a rule file in `.agents/rules/<agent-name>.md`
2. Create a workflow file in `.agents/workflows/<agent-name>.md`
3. Document the agent in this file

### Rule File Format

```markdown
## <agent-name>

Rules:
- Before using ListDir, Grep, or Read to explore any part of this codebase you are not already certain about, you MUST first run graphify query "<question>" (CLI) or query_graph (MCP, if available) if graphify-out/graph.json exists. This applies to implementation tasks, bug fixes, and refactors -- not only literal questions.
- Do NOT begin a task by listing directories or grepping broadly. Locate relevant files via graphify query / query_graph first, then read only the specific files it points to.
- Use graphify path "<A>" "<B>" / shortest_path (MCP) to understand relationships between two named components, and graphify explain "<concept>" / get_node (MCP) for a focused summary of one concept or module.
- These return a scoped subgraph, almost always far smaller than GRAPH_REPORT.md or raw grep output -- treat them as strictly cheaper than manual exploration.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review, or when query/path/explain do not surface enough context.
- Exception: once graphify has pointed you to specific files, reading their full contents directly is correct and expected -- the rule only governs how you locate code, not whether you read it once found.
- The graphify CLI is already installed and available directly on PATH. NEVER run it via npx, npm exec, or pip -- always call it as a plain 'graphify <command>' shell command.
- After modifying code, run graphify update . to keep the graph current (AST-only, no API cost).