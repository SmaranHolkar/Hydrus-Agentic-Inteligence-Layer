import asyncio
import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

class StdioMCPClient:
    """JSON-RPC client for external MCP servers running over stdio."""
    
    def __init__(self, name: str, command: str, args: List[str]):
        self.name = name
        self.command = command
        self.args = args
        self.process = None
        self.reader = None
        self.writer = None
        self.id_counter = 1
        self.pending_requests: Dict[int, asyncio.Future] = {}
        self.tools: List[Dict] = []
        self.read_task: Optional[asyncio.Task] = None
        
    async def start(self) -> bool:
        try:
            # Spawn stdio subprocess cleanly using asyncio
            self.process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            self.reader = self.process.stdout
            self.writer = self.process.stdin
            
            # Start background reader task
            self.read_task = asyncio.create_task(self._read_loop())
            
            # Handshake: initialize
            init_id = self.next_id()
            fut = asyncio.get_running_loop().create_future()
            self.pending_requests[init_id] = fut
            
            init_msg = {
                "jsonrpc": "2.0",
                "id": init_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "HydrusAgent", "version": "1.0.0"}
                }
            }
            await self._send(init_msg)
            await asyncio.wait_for(fut, timeout=10.0)
            
            # Handshake: initialized notification
            notify_msg = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            await self._send(notify_msg)
            
            # List tools to retrieve server capabilities
            list_id = self.next_id()
            list_fut = asyncio.get_running_loop().create_future()
            self.pending_requests[list_id] = list_fut
            
            list_msg = {
                "jsonrpc": "2.0",
                "id": list_id,
                "method": "tools/list",
                "params": {}
            }
            await self._send(list_msg)
            res = await asyncio.wait_for(list_fut, timeout=10.0)
            
            self.tools = res.get("tools", [])
            print(f"[ExternalMCP] Discovered {len(self.tools)} tools from server '{self.name}'")
            return True
            
        except Exception as e:
            print(f"[ExternalMCP] Failed to connect/handshake with '{self.name}': {e}")
            await self.stop()
            return False

    def next_id(self) -> int:
        self.id_counter += 1
        return self.id_counter

    async def _send(self, msg: Dict):
        data = json.dumps(msg) + "\n"
        self.writer.write(data.encode('utf-8'))
        await self.writer.drain()

    async def _read_loop(self):
        while True:
            try:
                line = await self.reader.readline()
                if not line:
                    break
                
                line_str = line.decode('utf-8').strip()
                if not line_str:
                    continue
                
                try:
                    msg = json.loads(line_str)
                except json.JSONDecodeError:
                    continue
                
                if "id" in msg:
                    msg_id = msg["id"]
                    if msg_id in self.pending_requests:
                        fut = self.pending_requests.pop(msg_id)
                        if "error" in msg:
                            fut.set_exception(Exception(msg["error"].get("message", "Unknown external error")))
                        else:
                            fut.set_result(msg.get("result", {}))
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[ExternalMCP] Error reading stdout for '{self.name}': {e}")
                break

    async def call_tool(self, tool_name: str, args: Dict) -> str:
        call_id = self.next_id()
        fut = asyncio.get_running_loop().create_future()
        self.pending_requests[call_id] = fut
        
        call_msg = {
            "jsonrpc": "2.0",
            "id": call_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": args
            }
        }
        await self._send(call_msg)
        try:
            result = await asyncio.wait_for(fut, timeout=30.0)
            
            # Format text outputs into Markdown summaries
            content_list = result.get("content", [])
            texts = []
            for item in content_list:
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return "\n".join(texts) or json.dumps(result)
        except Exception as e:
            return f"Error executing tool '{tool_name}' on server '{self.name}': {str(e)}"

    async def stop(self):
        if self.read_task:
            self.read_task.cancel()
            self.read_task = None
        if self.process:
            try:
                self.process.terminate()
                await self.process.wait()
            except:
                pass
            self.process = None


class ExternalMCPManager:
    """Manages discovery and lifecycle of external stdio servers via mcp_config.json."""
    
    def __init__(self, workspace: str):
        self.workspace = Path(workspace).resolve()
        self.config_path = self.workspace / "mcp_config.json"
        self.clients: Dict[str, StdioMCPClient] = {}
        self.initialized = False
        
    async def load_and_start_servers(self) -> Dict[str, StdioMCPClient]:
        if self.initialized:
            return self.clients
            
        if not self.config_path.exists():
            print(f"[ExternalMCP] No mcp_config.json found at '{self.config_path}'. External MCP disabled.")
            self.initialized = True
            return self.clients
            
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                
            servers_config = config.get("mcpServers", {})
            for name, item in servers_config.items():
                command = item.get("command")
                args = item.get("args", [])
                if not command:
                    continue
                    
                client = StdioMCPClient(name, command, args)
                success = await client.start()
                if success:
                    self.clients[name] = client
                    
        except Exception as e:
            print(f"[ExternalMCP] Failed loading config/starting servers: {e}")
            
        self.initialized = True
        return self.clients
        
    async def stop_all(self):
        for client in self.clients.values():
            await client.stop()
        self.clients.clear()
        self.initialized = False
