# Agentic Tooling

This section covers the places where AI agents meet GEECS. There are two
distinct strands, with different audiences, and it helps to keep them
separate in your head:

**Agents operating the lab.** The [GEECS MCP Server](../geecs_mcp/overview.md)
is a deployed lab service — the same standing as the gateway or the
console — that exposes GEECS-semantic operations (submit a scan, check its
progress, look up archived results, run post-scan analysis) as typed tools
an AI agent can call. An agent framework such as
[OSPREY](../geecs_mcp/osprey.md) connects to it and puts those tools in the
hands of an operator-facing assistant: "run the baseline analysis on
scan 12" becomes a tool call instead of a person clicking through a GUI.
If you are here to understand *what the AI assistant can actually do to the
machine, and what stops it doing more*, start with the
[MCP server overview](../geecs_mcp/overview.md).

**Agents developing the code.** [Skills](../skills/overview.md) are
instruction files for [Claude Code](https://claude.ai/code) sessions working
*on this repository* — codified workflows like landing a PR, triaging scan
logs, or repairing a Poetry environment. They never touch the beamline; they
make development on GEECS-Plugins repeatable and guarded.

The shared idea behind both strands is the same one that runs through the
whole suite: give the flexible, language-understanding layer (the agent) a
**well-defined contract** over a deterministic core, rather than free-form
access. For the lab that contract is the MCP server's tool surface and its
safety gates; for the codebase it is the skills' wrapped CLIs and rituals.
