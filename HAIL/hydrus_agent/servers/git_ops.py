import subprocess
import json
import os
from typing import List, Dict, Optional, Any
from pathlib import Path

class GitOpsServer:
    """MCP-compliant Git operations server."""
    
    def __init__(self, workspace: str):
        self.workspace = Path(workspace).resolve()
        self.git_available = self._check_git()
    
    def _check_git(self) -> bool:
        """Check if git is installed and workspace is a repo."""
        try:
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                return False
            
            # Check if workspace is a git repo
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _run_git(self, args: List[str], timeout: int = 30) -> Dict[str, Any]:
        """Execute git command in workspace."""
        if not self.git_available and args[0] != "init":
            return {"error": "Git not available or workspace is not a git repository"}
        
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {"error": f"Git command timed out after {timeout}s"}
        except Exception as e:
            return {"error": str(e)}
    
    # ─── TOOLS ────────────────────────────────────────────────────────
    
    def git_status(self) -> str:
        """Get current git status."""
        result = self._run_git(["status", "--short", "--branch"])
        return json.dumps(result, indent=2)
    
    def git_diff(self, staged: bool = False, file: Optional[str] = None) -> str:
        """Show code modifications."""
        args = ["diff"]
        if staged:
            args.append("--staged")
        if file:
            args.append(file)
        args.extend(["--no-color"])
        
        result = self._run_git(args)
        return json.dumps(result, indent=2)
    
    def git_add(self, files: List[str]) -> str:
        """Stage files for commit."""
        # Validate files are within workspace
        for f in files:
            try:
                f_path = (self.workspace / f).resolve()
                f_path.relative_to(self.workspace)
            except ValueError:
                return json.dumps({"error": f"File {f} is outside workspace"})
        
        result = self._run_git(["add"] + files)
        return json.dumps(result, indent=2)
    
    def git_commit(self, message: str, allow_empty: bool = False) -> str:
        """Commit staged changes."""
        args = ["commit", "-m", message]
        if allow_empty:
            args.append("--allow-empty")
        
        result = self._run_git(args)
        return json.dumps(result, indent=2)
    
    def git_log(self, max_count: int = 10, oneline: bool = False) -> str:
        """List recent revisions."""
        args = ["log", f"-n{max_count}", "--no-color"]
        if oneline:
            args.append("--oneline")
        else:
            args.extend(["--format=%H|%an|%ae|%ad|%s", "--date=short"])
        
        result = self._run_git(args)
        
        # Parse structured log if not oneline
        if not oneline and result.get("success") and result.get("stdout"):
            commits = []
            for line in result["stdout"].strip().split("\n"):
                if "|" in line:
                    parts = line.split("|", 4)
                    if len(parts) == 5:
                        commits.append({
                            "hash": parts[0],
                            "author": parts[1],
                            "email": parts[2],
                            "date": parts[3],
                            "message": parts[4]
                        })
            result["commits"] = commits
        
        return json.dumps(result, indent=2)
    
    def git_branch(self) -> str:
        """List branches and current branch."""
        result = self._run_git(["branch", "-v", "--no-color"])
        return json.dumps(result, indent=2)
    
    def git_init(self) -> str:
        """Initialize a git repository if one doesn't exist."""
        if self.git_available:
            return json.dumps({"error": "Git repository already exists"})
        
        result = self._run_git(["init"])
        self.git_available = True  # Recheck on next call
        return json.dumps(result, indent=2)
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "git_status",
                "description": "Get current git status (modified files, branch)",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "git_diff",
                "description": "Show code modifications. Can show staged changes or specific file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "staged": {"type": "boolean", "description": "Show staged changes", "default": False},
                        "file": {"type": "string", "description": "Specific file to diff"}
                    }
                }
            },
            {
                "name": "git_add",
                "description": "Stage files for commit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "files": {"type": "array", "items": {"type": "string"}, "description": "Files to stage"}
                    },
                    "required": ["files"]
                }
            },
            {
                "name": "git_commit",
                "description": "Commit staged changes with a message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Commit message"},
                        "allow_empty": {"type": "boolean", "description": "Allow empty commit", "default": False}
                    },
                    "required": ["message"]
                }
            },
            {
                "name": "git_log",
                "description": "List recent commits",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max_count": {"type": "integer", "description": "Max commits to show", "default": 10},
                        "oneline": {"type": "boolean", "description": "Compact format", "default": False}
                    }
                }
            },
            {
                "name": "git_branch",
                "description": "List branches",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "git_init",
                "description": "Initialize git repository (if not already initialized)",
                "parameters": {"type": "object", "properties": {}}
            }
        ]
