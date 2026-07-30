import os
import json
import time
from typing import List, Dict, Any

class Session:
    def __init__(self, name: str, workspace: str, hcl_instance: Any = None):
        self.name = name
        self.workspace = workspace
        self.hcl = hcl_instance
        self.chat_history: List[Dict[str, Any]] = []
        self.tool_history: List[Dict[str, Any]] = []
        self.permission_grants: Dict[str, float] = {}  # tool_name -> expiry timestamp
        self.variables: Dict[str, Any] = {}
        
        # Load existing session history if available
        self.sessions_dir = os.path.join(workspace, "data", "sessions")
        os.makedirs(self.sessions_dir, exist_ok=True)
        self.filepath = os.path.join(self.sessions_dir, f"{name}.json")
        self.load()
        
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        self.chat_history.append({
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time()
        })
        self.save()
        
        # Write episodic memory to HCL if active
        if self.hcl is not None and role == "observation":
            try:
                summary = f"Tool result observation: {content[:200]}"
                if hasattr(self.hcl, "SML") and hasattr(self.hcl, "model") and hasattr(self.hcl, "tokenizer"):
                    from hcl_core.hcl import embed, NodeType
                    query_vector = embed(summary, self.hcl.model, self.hcl.tokenizer)
                    payload = {
                        "id": f"agent_tool_{int(time.time())}",
                        "type": NodeType.EPISODIC.value,
                        "summary": summary,
                        "cluster_id": 98,
                        "session": self.name
                    }
                    self.hcl.SML.write(query_vector, confidence=1.0, payload=payload)
            except Exception as e:
                print(f"[Session] Failed to write episodic log to HCL: {e}")

    def add_tool_call(self, tool_name: str, args: dict, result: str):
        self.tool_history.append({
            "tool": tool_name,
            "args": args,
            "result": result[:500] if isinstance(result, str) else str(result),
            "timestamp": time.time()
        })
        self.save()

    def get_chat_history(self) -> List[Dict[str, Any]]:
        return self.chat_history

    def get_permission_state(self, tool_name: str) -> bool:
        """Check if tool is pre-approved for this session."""
        if tool_name in self.permission_grants:
            if time.time() < self.permission_grants[tool_name]:
                return True
        return False

    def grant_permission(self, tool_name: str, duration_seconds: int = 3600):
        """Grant session-wide temporary permission for a tool."""
        self.permission_grants[tool_name] = time.time() + duration_seconds

    def save(self):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump({
                    "name": self.name,
                    "chat_history": self.chat_history,
                    "tool_history": self.tool_history,
                    "permission_grants": self.permission_grants,
                    "variables": self.variables
                }, f, indent=4)
        except Exception as e:
            print(f"[Session] Save failed: {e}")

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.chat_history = data.get("chat_history", [])
                    self.tool_history = data.get("tool_history", [])
                    self.permission_grants = data.get("permission_grants", {})
                    self.variables = data.get("variables", {})
            except Exception as e:
                print(f"[Session] Load failed: {e}")
