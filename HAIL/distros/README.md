# HAIL Distributions (Distros)

Following the Linux distribution model, HAIL provides a sacred minimal kernel (`hail_core`) while allowing different packaging targets ("distros") for distinct user personas.

| Distro | Persona | Description | Key Capabilities |
| :--- | :--- | :--- | :--- |
| **HAIL Web** | Casual Users | Lightweight browser bundle powered by WASM | Zero install, in-memory lattice, local storage persistence |
| **HAIL Desktop** | Daily & Power Users | Full GPU acceleration desktop wrapper | File system access, local LLM runners (Ollama/llama.cpp), MCP tools |
| **HAIL Server** | Teams & Infrastructure | Multi-tenant headless service | Dedicated server API, shared skills, admin role control |

## Monorepo Layout

- `src/hail_core/` — Sacred Kernel (Memory Lattice, HydrusOpt, Skill Engine)
- `server_api/` — Reference implementation for **HAIL Server Distro**
- `robot_sim/` — Example GUI / Simulation Distro tool
- `hail-skills/` — Modular Skill Package Repository
