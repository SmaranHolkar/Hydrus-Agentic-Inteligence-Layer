# HAIL Skill Ecosystem

This directory contains skills for the HAIL AI platform, structured according to the Linux-inspired governance model.

## Directory Structure

```
hail-skills/
├── official/           # Vivere Labs maintained & security-audited core skills
├── community/          # Peer-reviewed community skills
└── experimental/       # Sandbox and wild-west skills (user beware)
```

## Creating a Skill

1. Create a subfolder in `hail-skills/community/<your-skill-name>/` (or `experimental/`).
2. Add a `manifest.json` conforming to `schema.json`:
   ```json
   {
     "name": "my-custom-skill",
     "version": "0.1.0",
     "description": "Short description of what the skill does",
     "author": "Your Name",
     "tier": "community",
     "permissions": ["memory:read", "memory:write"],
     "memory_strata": ["surface"],
     "entrypoint": "main.py"
   }
   ```
3. Implement your entrypoint script.
4. Load into HAIL Core using `hail.skills.discover_and_load_all()`.
