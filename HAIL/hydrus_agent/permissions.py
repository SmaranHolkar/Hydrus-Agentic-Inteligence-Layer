from abc import ABC, abstractmethod
import asyncio

class PermissionGate(ABC):
    @abstractmethod
    async def request(self, action: str, danger_level: str, details: dict) -> bool:
        """Request permission. Returns True (or 'always') if granted."""
        pass

class CLIPermissionGate(PermissionGate):
    """Interactive CLI permission prompts using Rich."""
    
    async def request(self, action: str, danger_level: str, details: dict) -> bool:
        from rich.console import Console
        from rich.panel import Panel
        
        console = Console()
        color = "red" if danger_level == "high" else "yellow" if danger_level == "medium" else "green"
        
        console.print(Panel(
            f"[bold {color}]⚠️  Permission Required[/bold {color}]\n\n"
            f"[bold]Action:[/bold] {action}\n"
            f"[bold]Risk Level:[/bold] {danger_level.upper()}\n"
            f"[bold]Details:[/bold] {details}",
            border_style=color
        ))
        
        # Use asyncio to not block the thread execution
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: console.input("[bold]Allow? [y/N/always]: [/bold]").strip().lower()
        )
        
        if response == "always":
            return "always"  # Special signal for session-wide grant
        return response in ("y", "yes")

class AutoAllowGate(PermissionGate):
    """Auto-allows low/medium, blocks high. Useful for tests or non-interactive environments."""
    
    async def request(self, action: str, danger_level: str, details: dict) -> bool:
        if danger_level == "high":
            return False
        return True

class APIPermissionGate(PermissionGate):
    """For API/Web environment. High-danger requires UI confirmation."""
    def __init__(self):
        self.pending_requests = {}
        self.responses = {}
        
    async def request(self, action: str, danger_level: str, details: dict) -> bool:
        if danger_level == "low":
            return True
            
        # For medium/high, we yield a permission request to the client
        # In a multi-user/web server, we wait for a client response
        # Here we can create a unique request ID
        req_id = details.get("req_id")
        if not req_id:
            import uuid
            req_id = str(uuid.uuid4())
        self.pending_requests[req_id] = {
            "action": action,
            "danger_level": danger_level,
            "details": details,
            "resolved": False
        }
        
        # In actual API streaming, we emit the permission event and poll or wait for an event trigger.
        # We can implement a non-blocking timeout wait
        timeout = 60 # 60 seconds timeout
        elapsed = 0
        while not self.pending_requests[req_id]["resolved"] and elapsed < timeout:
            await asyncio.sleep(0.5)
            elapsed += 0.5
            
        if self.pending_requests[req_id]["resolved"]:
            return self.responses.get(req_id, False)
        return False
        
    def resolve(self, req_id: str, approved: bool):
        if req_id in self.pending_requests:
            self.pending_requests[req_id]["resolved"] = True
            self.responses[req_id] = approved
