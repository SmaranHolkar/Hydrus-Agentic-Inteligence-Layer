"""
Community Skill Entrypoint Template.
Invoked by HAIL Core runtime when the skill is active.
"""

def execute(hail_kernel, payload=None):
    print(f"[Skill Execution] Active context payload: {payload}")
    # Write memory to surface lattice
    addr = hail_kernel.write([0.1] * 64, confidence=0.9, payload={"source": "my-custom-skill"})
    return {"status": "success", "memory_address": addr}

if __name__ == "__main__":
    print("Run this skill through HAIL Core or `hail.skills.load_from_path(...)`")
