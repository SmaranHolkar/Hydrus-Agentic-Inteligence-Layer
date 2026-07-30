from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Any
import time
import json

class EventType(Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    PERMISSION = "permission"
    ERROR = "error"
    SAFETY_REVIEW = "safety"
    COMPLETE = "complete"

@dataclass
class AgentEvent:
    type: EventType
    content: str
    metadata: Optional[dict] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}
    
    def to_json(self) -> str:
        return json.dumps({
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata or {},
            "timestamp": self.timestamp
        }, default=str)
    
    @classmethod
    def thought(cls, content: str) -> "AgentEvent":
        return cls(EventType.THOUGHT, content)
    
    @classmethod
    def action(cls, tool: str, args: dict) -> "AgentEvent":
        return cls(EventType.ACTION, f"Using {tool}", {"tool": tool, "args": args})
    
    @classmethod
    def observation(cls, content: str, tool: str = None) -> "AgentEvent":
        return cls(EventType.OBSERVATION, content, {"tool": tool})
    
    @classmethod
    def error(cls, content: str, recoverable: bool = True) -> "AgentEvent":
        return cls(EventType.ERROR, content, {"recoverable": recoverable})
    
    @classmethod
    def permission_request(cls, action: str, danger_level: str, details: dict) -> "AgentEvent":
        return cls(EventType.PERMISSION, action, {
            "danger_level": danger_level,
            "details": details
        })
    
    @classmethod
    def complete(cls, answer: str) -> "AgentEvent":
        return cls(EventType.COMPLETE, answer)
