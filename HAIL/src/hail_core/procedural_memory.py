import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class SkillTrace:
    tool_name: str
    last_args: Dict[str, Any]
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    known_failure_modes: List[str] = field(default_factory=list)
    blacklisted_until_turn: int = 0

class ProceduralMemoryManager:
    """
    Living Procedural Memory manager.
    Tracks execution history, failure modes, and enforces Execution Backoff if tools fail repeatedly.
    """
    def __init__(self, failure_threshold: int = 3, backoff_duration_turns: int = 5):
        self.failure_threshold = failure_threshold
        self.backoff_duration_turns = backoff_duration_turns
        self.traces: Dict[str, SkillTrace] = {}
        self.current_turn = 0

    def advance_turn(self):
        self.current_turn += 1

    def record_execution(self, tool_name: str, args: Dict[str, Any], success: bool, error: Optional[str] = None):
        if tool_name not in self.traces:
            self.traces[tool_name] = SkillTrace(tool_name=tool_name, last_args=args)

        trace = self.traces[tool_name]
        trace.last_args = args

        if success:
            trace.success_count += 1
            trace.consecutive_failures = 0
        else:
            trace.failure_count += 1
            trace.consecutive_failures += 1
            trace.last_error = error
            if error and error not in trace.known_failure_modes:
                trace.known_failure_modes.append(error)

            if trace.consecutive_failures >= self.failure_threshold:
                trace.blacklisted_until_turn = self.current_turn + self.backoff_duration_turns

    def is_blacklisted(self, tool_name: str) -> Tuple[bool, str]:
        """
        Check if a tool is temporarily backoff-blacklisted due to repeated failures.
        """
        if tool_name not in self.traces:
            return False, "Tool is clean."

        trace = self.traces[tool_name]
        if trace.blacklisted_until_turn > self.current_turn:
            remaining = trace.blacklisted_until_turn - self.current_turn
            return True, f"Tool '{tool_name}' is in execution backoff after {trace.consecutive_failures} consecutive failures. Backoff active for {remaining} more turn(s). Last error: {trace.last_error}"

        return False, "Tool active."

    def get_skill_hint(self, tool_name: str) -> Optional[str]:
        """
        Returns procedural advice learned from past executions.
        """
        if tool_name not in self.traces:
            return None

        trace = self.traces[tool_name]
        if trace.known_failure_modes:
            return f"Procedural Memory Note for {tool_name}: Solved {trace.success_count}x, failed {trace.failure_count}x. Known pitfalls: {'; '.join(trace.known_failure_modes[-2:])}"
        return f"Procedural Memory Note for {tool_name}: Solved {trace.success_count}x successfully."
