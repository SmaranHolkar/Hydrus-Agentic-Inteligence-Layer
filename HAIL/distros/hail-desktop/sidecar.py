"""
HAIL Desktop — Tauri Python Sidecar IPC Bridge.
Communicates with Tauri Rust host via JSON-RPC over stdin/stdout.
Exposes hail_core memory lattice, skills loader, and local model runner operations.
"""

import sys
import json
import os
from pathlib import Path

# Add kernel source path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hail_core import HAIL, HAILConfig, ModelConfig

class HailSidecarBridge:
    def __init__(self):
        storage_path = Path(__file__).parent / "desktop_memory.hcl"
        skills_dir = Path(__file__).resolve().parents[2] / "hail-skills"
        
        config = HAILConfig(
            storage_path=storage_path,
            skills_dir=skills_dir,
            autosave=True
        )
        self.hail = HAIL(config)

    def process_command(self, cmd_data: dict) -> dict:
        if not isinstance(cmd_data, dict):
            return {"error": "Invalid command format: payload must be a JSON object"}

        action = cmd_data.get("action")
        
        if action == "status":
            return {
                "status": "online",
                "distro": "HAIL Desktop (Tauri)",
                "kernel": "HAIL Core v0.1",
                "skills_count": len(self.hail.skills.list_skills()),
                "storage_path": str(self.hail.config.storage_path)
            }
        elif action == "list_skills":
            return {
                "skills": self.hail.skills.list_skills()
            }
        elif action == "write_memory":
            raw_text = str(cmd_data.get("text", ""))[:5000]  # Cap input text length
            try:
                confidence = float(cmd_data.get("confidence", 0.85))
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 0.85

            # Create a mock 64-dim embedding or store payload
            addr = self.hail.write([0.1] * 64, confidence=confidence, payload={"text": raw_text})
            return {"status": "success", "address": addr, "text": raw_text}
        elif action == "recall_memory":
            try:
                raw_k = int(cmd_data.get("k", 5))
                k = max(1, min(raw_k, 100))  # Enforce reasonable bounds for k
            except (ValueError, TypeError):
                k = 5

            results = self.hail.recall([0.1] * 64, k=k)
            return {"status": "success", "results": results}
        else:
            return {"error": f"Unknown action: {action}"}

def main():
    bridge = HailSidecarBridge()

    # If launched with a single argument JSON string (CLI testing mode)
    if len(sys.argv) > 1:
        try:
            req = json.loads(sys.argv[1])
            res = bridge.process_command(req)
            print(json.dumps(res))
            return
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            return

    # Continuous stdin loop for Tauri Sidecar IPC
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            res = bridge.process_command(req)
            print(json.dumps(res))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
