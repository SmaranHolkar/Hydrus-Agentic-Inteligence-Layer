"""
Skill Manifest specification for HAIL Kernel.
Defines metadata, permissions, memory strata access, and entry points for skills.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

VALID_TIERS = {"official", "community", "experimental"}
VALID_PERMISSIONS = {
    "memory:read",
    "memory:write",
    "memory:abyss",
    "fs:read",
    "fs:write",
    "network:outbound",
    "mcp:connect",
}

@dataclass
class SkillManifest:
    name: str
    version: str
    description: str
    author: str = "Unknown"
    tier: str = "experimental"
    permissions: List[str] = field(default_factory=list)
    memory_strata: List[str] = field(default_factory=lambda: ["surface"])
    mcp_connectors: List[str] = field(default_factory=list)
    entrypoint: str = "main.py"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "tier": self.tier,
            "permissions": self.permissions,
            "memory_strata": self.memory_strata,
            "mcp_connectors": self.mcp_connectors,
            "entrypoint": self.entrypoint,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillManifest":
        return cls(
            name=data.get("name", ""),
            version=data.get("version", "0.1.0"),
            description=data.get("description", ""),
            author=data.get("author", "Unknown"),
            tier=data.get("tier", "experimental"),
            permissions=data.get("permissions", []),
            memory_strata=data.get("memory_strata", ["surface"]),
            mcp_connectors=data.get("mcp_connectors", []),
            entrypoint=data.get("entrypoint", "main.py"),
            metadata=data.get("metadata", {}),
        )
