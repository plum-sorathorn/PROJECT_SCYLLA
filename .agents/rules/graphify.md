---
trigger: always_on
description: Consult the graphify knowledge graph at graphify-out/ for codebase and architecture questions.
---

## graphify

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