# HAIL Skill Boilerplate Template

Use this template directory to create your own custom skills for the HAIL AI ecosystem.

## How to use:

1. Copy this `template` directory into `hail-skills/community/<your-skill-name>/` or `hail-skills/experimental/<your-skill-name>/`.
2. Edit `manifest.json`:
   - Set a unique `name`.
   - Choose allowed `permissions` (e.g. `memory:read`, `memory:write`).
3. Implement your custom logic in `main.py`.
4. Test loading into HAIL:
   ```python
   from hail_core import HAIL
   hail = HAIL()
   manifest = hail.skills.load_from_path("hail-skills/community/<your-skill-name>")
   print(f"Loaded skill: {manifest.name}")
   ```
