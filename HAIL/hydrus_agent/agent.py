import asyncio
import json
from typing import AsyncGenerator, List, Dict, Any, Tuple
from .events import AgentEvent, EventType
from .llm_bridge import HydrusOptAdapter
from .mcp_bus import MCPBus
from .session import Session
from .security import WorkspaceGuard
from .permissions import PermissionGate, CLIPermissionGate
from .cognitive_gateway import CognitiveMemoryGateway

class HydrusAgent:
    def __init__(self, workspace: str, model_adapter: HydrusOptAdapter, 
                 permission_gate: PermissionGate = None, hcl_instance: Any = None):
        self.workspace = workspace
        self.llm = model_adapter
        self.bus = MCPBus(workspace)
        self.guard = WorkspaceGuard(workspace)
        self.permissions = permission_gate or CLIPermissionGate()
        self.hcl = hcl_instance
        self.cmg = CognitiveMemoryGateway(workspace)
        self.max_steps = 15
        
    async def run_stream(self, prompt: str, session: Session = None) -> AsyncGenerator[AgentEvent, None]:
        session = session or Session("default", self.workspace, self.hcl)
        session.add_message("user", prompt)
        self.cmg.procedural.advance_turn()

        
        # Load external servers and map tools dynamically
        try:
            await self.bus.load_external_servers()
        except Exception as e:
            print(f"[HydrusAgent] Failed to load external MCP servers: {e}")
            
        tools = self.bus.get_tool_definitions()
        
        # 1. Retrieve memory context from Stratified Memory Lattice (SML)
        context = ""
        if self.hcl is not None:
            try:
                from hcl_core.hcl import embed
                # Calculate prompt embedding and query SML
                query_vector = embed(prompt, self.llm.model, self.llm.tokenizer)
                memories = self.hcl.retrieve(query_vector, k=3)
                context = self.hcl.format_injection(memories)
                if context.strip():
                    yield AgentEvent(EventType.SAFETY_REVIEW, f"HCL Memory Grounding Injected:\n{context}")
            except Exception as e:
                print(f"[HydrusAgent] HCL retrieve failed: {e}")
        
        yield AgentEvent.thought("Agent starting ReAct loop. Formulating initial thoughts...")
        
        for step in range(self.max_steps):
            # Format chat history and inject context
            history = session.get_chat_history()
            react_prompt = self.llm.build_react_prompt(prompt, tools, history, workspace=self.workspace, context=context)
                
            # Call local LLM to generate next step
            raw_output = self.llm.generate(react_prompt)
            
            # Parse thoughts, action, and arguments
            thought, action_name, action_args = self.llm.parse_react_output(raw_output)
            
            if thought:
                yield AgentEvent.thought(thought)
                session.add_message("assistant", f"Thought: {thought}")
                
            # If no action found, report error
            if not action_name:
                yield AgentEvent.error("Failed to parse Action block from model output. Retrying with explicit prompt guide...", recoverable=True)
                session.add_message("observation", "Error: No <action> block found. Please write: <action>ToolName|{\"arg\":\"val\"}</action>")
                continue
                
            # Complete Final Answer?
            if action_name == "FinalAnswer":
                answer = action_args.get("answer", "Task complete")
                yield AgentEvent.complete(answer)
                session.add_message("assistant", f"Final Answer: {answer}")
                return
                
            # Report Action Call
            yield AgentEvent.action(action_name, action_args)
            
            # 1.5 Procedural Memory Tool Execution Backoff Check
            is_blacklisted, backoff_reason = self.cmg.procedural.is_blacklisted(action_name)
            if is_blacklisted:
                yield AgentEvent.error(f"CMG Execution Backoff: {backoff_reason}", recoverable=True)
                session.add_message("observation", f"Error: Tool execution blocked by Procedural Backoff Policy. {backoff_reason}")
                continue

            # 2. Safety Validation Gate
            is_safe, danger, reason = self._validate_action(action_name, action_args)
            if not is_safe:
                yield AgentEvent.error(f"HCL Safety Policy blocked Action: {reason}", recoverable=False)
                session.add_message("observation", f"Error: Action blocked by security. Reason: {reason}")
                return
                
            # 3. Permission Consent Gate
            # Skip permission prompt if session has marked tool as pre-approved
            if danger in ("medium", "high"):
                if session.get_permission_state(action_name):
                    yield AgentEvent(EventType.SAFETY_REVIEW, f"Tool '{action_name}' is pre-approved for this session.")
                else:
                    import uuid
                    req_id = str(uuid.uuid4())
                    details_with_id = {"tool": action_name, "args": action_args, "req_id": req_id}
                    
                    yield AgentEvent.permission_request(
                        action=f"{action_name}({json.dumps(action_args)})",
                        danger_level=danger,
                        details=details_with_id
                    )
                    
                    permitted = await self.permissions.request(
                        action=f"{action_name}({json.dumps(action_args)})",
                        danger_level=danger,
                        details=details_with_id
                    )
                    
                    if not permitted:
                        yield AgentEvent.error(f"Permission denied by user for action {action_name}", recoverable=True)
                        session.add_message("observation", f"Error: Permission denied by user for {action_name}. Please try an alternative approach.")
                        continue
                    elif permitted == "always":
                        # Grant session-wide auto-allow
                        session.grant_permission(action_name)
                        yield AgentEvent(EventType.SAFETY_REVIEW, f"Granted temporary session-wide permission for tool: {action_name}")
            
            # 4. Tool Execution & Cognitive Memory Gateway (CMG) Pipeline
            try:
                yield AgentEvent(EventType.SAFETY_REVIEW, f"Running tool: {action_name}...")
                # Run the tool via the MCP Bus
                raw_result = await self.bus.execute(action_name, action_args)
                obs_text = str(raw_result)

                # Route through Cognitive Memory Gateway (Sanitization, Firewall, Grounding)
                cmg_res = await self.cmg.process_external_output(action_name, action_args, obs_text)

                if cmg_res.get("quarantined"):
                    warn = f"[CMG Firewall Quarantined Input]: {cmg_res.get('quarantine_reason')}"
                    yield AgentEvent.error(warn, recoverable=True)
                    session.add_message("observation", f"Error: {warn}")
                    continue

                obs = cmg_res["working_context"]
                # Truncate large tool outputs
                if len(obs) > 10000:
                    obs = obs[:10000] + "\n... [Content truncated due to size limits]"
                    
                yield AgentEvent.observation(obs, action_name)
                session.add_message("observation", obs)
                session.add_tool_call(action_name, action_args, obs)
                
            except Exception as e:
                err_msg = f"Error: Tool execution crashed. {str(e)}"
                self.cmg.procedural.record_execution(action_name, action_args, success=False, error=str(e))
                yield AgentEvent.error(err_msg, recoverable=True)
                session.add_message("observation", err_msg)
                
        yield AgentEvent.error("Agent reached maximum execution steps (15) without solving the task.", recoverable=False)

    def _validate_action(self, name: str, args: dict) -> Tuple[bool, str, str]:
        """Validate path or commands dynamically before dispatching to MCP Bus."""
        if name == "run_command" and "command" in args:
            return self.guard.validate_command(args["command"])
        if name in ("read_file", "write_file", "patch_file", "delete_file", "make_directory", "delete_directory", "list_directory") and "path" in args:
            try:
                self.guard.validate_path(args["path"])
                return True, "low", "Path resides inside workspace boundaries."
            except PermissionError as e:
                return False, "high", str(e)
        return True, "low", "Generic verification passed."
