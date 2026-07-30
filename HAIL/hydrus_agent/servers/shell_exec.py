import os
import subprocess
from typing import Dict, Any
from ..security import WorkspaceGuard

class ShellExecServer:
    def __init__(self, guard: WorkspaceGuard):
        self.guard = guard
        self.workspace = str(guard.root)
        self.max_output = 5000

    def run_command(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Execute terminal shell commands inside the workspace directory.
        Restricts destructive commands via WorkspaceGuard and caps stdout/stderr.
        """
        # Validate command safety through security guard first
        is_allowed, danger_level, reason = self.guard.validate_command(command)
        if not is_allowed:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Error: Command blocked by WorkspaceGuard. Reason: {reason}"
            }
            
        env = os.environ.copy()
        # On non-Windows platforms, restrict path for security.
        # On Windows, keep PATH so developers can run local git, python, and pytest executables.
        if os.name != 'nt':
            env["PATH"] = "/usr/bin:/bin"
        env.pop("SUDO_ASKPASS", None)
        
        try:
            # Execute command inside workspace root directory
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            stdout = proc.stdout
            if len(stdout) > self.max_output:
                stdout = stdout[:self.max_output] + f"\n... [truncated {len(proc.stdout) - self.max_output} characters]"
                
            stderr = proc.stderr
            if len(stderr) > self.max_output:
                stderr = stderr[:self.max_output] + f"\n... [truncated {len(proc.stderr) - self.max_output} characters]"
                
            return {
                "returncode": proc.returncode,
                "stdout": stdout,
                "stderr": stderr
            }
        except subprocess.TimeoutExpired:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Error: Command execution timed out after {timeout} seconds."
            }
        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Error: Subprocess launch failed. {str(e)}"
            }
