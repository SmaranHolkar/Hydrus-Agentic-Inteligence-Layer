import os
from typing import List, Dict, Any
from .security import WorkspaceGuard
from .servers.file_system import FileSystemServer
from .servers.shell_exec import ShellExecServer
from .servers.git_ops import GitOpsServer
from .external_mcp import ExternalMCPManager
from .servers.browser import BrowserServer
from .servers.search import SearchServer

class MCPBus:
    def __init__(self, workspace: str, session_name: str = "default"):
        self.workspace = workspace
        self.guard = WorkspaceGuard(workspace)
        self.fs_server = FileSystemServer(self.guard, session_name)
        self.shell_server = ShellExecServer(self.guard)
        self.git_server = GitOpsServer(workspace)
        self.external_manager = ExternalMCPManager(workspace)
        self.browser_server = BrowserServer(self.guard, session_name)
        self.search_server = SearchServer(self.guard, session_name)
        
        # We will also register a basic in-memory web/weather tool fallback here
        self.tools = {
            "read_file": {
                "description": "Read file contents, with optional line range constraint.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute or relative path to file"},
                        "start_line": {"type": "integer", "description": "1-indexed starting line to view"},
                        "end_line": {"type": "integer", "description": "1-indexed ending line to view"}
                    },
                    "required": ["path"]
                },
                "handler": self.fs_server.read_file
            },
            "write_file": {
                "description": "Write entire contents to a file (creates parent directories).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to file"},
                        "content": {"type": "string", "description": "File contents"}
                    },
                    "required": ["path", "content"]
                },
                "handler": self.fs_server.write_file
            },
            "patch_file": {
                "description": "Replace a specific target text block with a replacement text block in a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to file"},
                        "old_string": {"type": "string", "description": "Exact text block to replace"},
                        "new_string": {"type": "string", "description": "Replacement text block"}
                    },
                    "required": ["path", "old_string", "new_string"]
                },
                "handler": self.fs_server.patch_file
            },
            "make_directory": {
                "description": "Create a new directory (folder) at the specified path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative directory path (e.g. 'hello' or 'tests/hello')"}
                    },
                    "required": ["path"]
                },
                "handler": self.fs_server.make_directory
            },
            "delete_file": {
                "description": "Delete/remove a file at the specified path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to delete"}
                    },
                    "required": ["path"]
                },
                "handler": self.fs_server.delete_file
            },
            "delete_directory": {
                "description": "Delete/remove an entire directory (folder) at the specified path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the directory to delete"}
                    },
                    "required": ["path"]
                },
                "handler": self.fs_server.delete_directory
            },
            "list_directory": {
                "description": "List all files and subdirectories inside a directory path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path", "default": "."}
                    }
                },
                "handler": self.fs_server.list_directory
            },
            "grep_search": {
                "description": "Find text or regex pattern matches recursively within workspace files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Literal search string or regex pattern"},
                        "path": {"type": "string", "description": "Directory to search in", "default": "."}
                    },
                    "required": ["query"]
                },
                "handler": self.fs_server.grep_search
            },
            "run_command": {
                "description": "Run shell command inside workspace directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"}
                    },
                    "required": ["command"]
                },
                "handler": self.shell_server.run_command
            },
            "web_search": {
                "description": "Query the web for search engine listings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query terms"}
                    },
                    "required": ["query"]
                },
                "handler": self.search_server.web_search
            },
            "fetch_webpage": {
                "description": "Fetch HTML content of a URL and parse it into readable Markdown.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "HTTP URL to fetch"},
                        "query": {"type": "string", "description": "Search query terms to apply BM25 content filtering"}
                    },
                    "required": ["url"]
                },
                "handler": self.search_server.fetch_webpage
            },
            "web_navigate": {
                "description": "Load a webpage. Returns the page text and a list of numbered interactive elements.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The HTTP/HTTPS URL of the website to visit"}
                    },
                    "required": ["url"]
                },
                "handler": self.browser_server.web_navigate
            },
            "web_click": {
                "description": "Click on a numbered interactive link or button (form submission).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "The 1-indexed number of the element to click"}
                    },
                    "required": ["index"]
                },
                "handler": self.browser_server.web_click
            },
            "web_type": {
                "description": "Type text into a numbered text input field.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "The 1-indexed number of the input field"},
                        "text": {"type": "string", "description": "The string to type into the field"}
                    },
                    "required": ["index", "text"]
                },
                "handler": self.browser_server.web_type
            },
            "web_back": {
                "description": "Go back to the previous webpage in browser history.",
                "parameters": {"type": "object", "properties": {}},
                "handler": self.browser_server.web_back
            },
            "git_status": {
                "description": "Get current git status (modified files, branch)",
                "parameters": {"type": "object", "properties": {}},
                "handler": self.git_server.git_status
            },
            "git_diff": {
                "description": "Show code modifications. Can show staged changes or specific file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "staged": {"type": "boolean", "description": "Show staged changes", "default": False},
                        "file": {"type": "string", "description": "Specific file to diff"}
                    }
                },
                "handler": self.git_server.git_diff
            },
            "git_add": {
                "description": "Stage files for commit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "files": {"type": "array", "items": {"type": "string"}, "description": "Files to stage"}
                    },
                    "required": ["files"]
                },
                "handler": self.git_server.git_add
            },
            "git_commit": {
                "description": "Commit staged changes with a message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Commit message"},
                        "allow_empty": {"type": "boolean", "description": "Allow empty commit", "default": False}
                    },
                    "required": ["message"]
                },
                "handler": self.git_server.git_commit
            },
            "git_log": {
                "description": "List recent commits",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max_count": {"type": "integer", "description": "Max commits to show", "default": 10},
                        "oneline": {"type": "boolean", "description": "Compact format", "default": False}
                    }
                },
                "handler": self.git_server.git_log
            },
            "git_branch": {
                "description": "List branches",
                "parameters": {"type": "object", "properties": {}},
                "handler": self.git_server.git_branch
            },
            "git_init": {
                "description": "Initialize git repository (if not already initialized)",
                "parameters": {"type": "object", "properties": {}},
                "handler": self.git_server.git_init
            }
        }
        
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        definitions = []
        for name, info in self.tools.items():
            definitions.append({
                "name": name,
                "description": info["description"],
                "parameters": info["parameters"]
            })
        return definitions

    async def load_external_servers(self):
        """Discovers and starts external stdio servers dynamically mapping their tools."""
        clients = await self.external_manager.load_and_start_servers()
        for name, client in clients.items():
            for tool in client.tools:
                tool_name = tool["name"]
                if tool_name in self.tools:
                    print(f"[MCPBus] Warning: Collision avoided. Native tool '{tool_name}' overrides external tool.")
                    continue
                    
                def make_handler(c=client, tn=tool_name):
                    async def handle(**kwargs):
                        return await c.call_tool(tn, kwargs)
                    return handle
                    
                self.tools[tool_name] = {
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                    "handler": make_handler()
                }

    async def shutdown(self):
        """Gracefully stop all external MCP servers."""
        await self.external_manager.stop_all()

    async def execute(self, tool_name: str, args: Dict[str, Any]) -> Any:
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found.")
        
        handler = self.tools[tool_name]["handler"]
        
        import inspect
        sig = inspect.signature(handler)
        valid_args = {}
        
        # Strip or map parameters based on function signature
        for param_name, param in sig.parameters.items():
            if param_name in args:
                valid_args[param_name] = args[param_name]
            elif param.default == inspect.Parameter.empty and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                # Fallback mapping if required parameter is missing but raw value was parsed
                if param_name == "path" and "raw" in args:
                    valid_args["path"] = args["raw"]
                elif param_name == "command" and "raw" in args:
                    valid_args["command"] = args["raw"]
                elif param_name == "query" and "raw" in args:
                    valid_args["query"] = args["raw"]
                elif param_name == "url" and "raw" in args:
                    valid_args["url"] = args["raw"]
        
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_kwargs:
            valid_args = args
            
        # Execute tool
        if inspect.iscoroutinefunction(handler):
            return await handler(**valid_args)
        else:
            return handler(**valid_args)

    def _web_search_fallback(self, query: str) -> str:
        """Call standard web search logic safely."""
        from server_api.agent import web_search_func
        res = web_search_func(query)
        return res

    def _fetch_webpage_fallback(self, url: str) -> str:
        """Fetch URL and clean to Markdown text."""
        import urllib.request
        import re
        
        # Safe URL check
        if not (url.startswith("http://") or url.startswith("https://")):
            return "Error: Invalid protocol. Only http/https supported."
            
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as response:
                html = response.read().decode('utf-8', errors='ignore')
                
                # Simple HTML to markdown parser
                # Strip style/script
                html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
                # Headings
                html = re.sub(r'<h[1-6][^>]*>(.*?)</h[1-6]>', r'\n\n## \1\n', html, flags=re.IGNORECASE)
                # Links
                html = re.sub(r'<a[^>]+href=["\'](https?://[^"\']+)["\'][^>]*>(.*?)</a>', r'[\2](\1)', html, flags=re.IGNORECASE)
                # Strip other tags
                text = re.sub(r'<[^>]+>', ' ', html)
                # Clean spacing
                text = re.sub(r'\s+', ' ', text).strip()
                # Line breaks on paragraphs
                text = text.replace("##", "\n\n## ")
                
                # Truncate
                if len(text) > 10000:
                    text = text[:10000] + "\n\n... [Content truncated due to size]"
                return text
        except Exception as e:
            return f"Error: Failed to fetch webpage. {str(e)}"
