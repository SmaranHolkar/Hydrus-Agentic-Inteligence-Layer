import os
import time
import shutil
import re
from pathlib import Path
from typing import Dict, Any, Optional
from ..security import WorkspaceGuard

class FileSystemServer:
    def __init__(self, guard: WorkspaceGuard, session_name: str = "default"):
        self.guard = guard
        self.session_name = session_name
        self.max_output = 10000

    def truncate_result(self, result: str) -> str:
        if len(result) > self.max_output:
            return result[:self.max_output] + f"\n\n... [truncated {len(result) - self.max_output} characters]"
        return result

    def read_file(self, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        try:
            validated = self.guard.validate_path(path)
            if not validated.exists():
                return f"Error: File '{path}' does not exist."
            if validated.is_dir():
                return f"Error: '{path}' is a directory, use list_directory instead."
                
            with open(validated, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                
            if start_line is not None or end_line is not None:
                start = max(1, start_line) if start_line is not None else 1
                end = min(len(lines), end_line) if end_line is not None else len(lines)
                if start > end:
                    return f"Error: start_line ({start}) is greater than end_line ({end})."
                slice_lines = lines[start-1:end]
                result = "".join(slice_lines)
                header = f"--- Reading '{path}' (Lines {start}-{end} of {len(lines)}) ---\n"
                return self.truncate_result(header + result)
            else:
                result = "".join(lines)
                header = f"--- Reading '{path}' (Entire File, {len(lines)} lines) ---\n"
                return self.truncate_result(header + result)
        except Exception as e:
            return f"Error: Failed to read file '{path}'. {str(e)}"

    def write_file(self, path: str, content: str) -> str:
        try:
            validated = self.guard.validate_path(path)
            
            # Make sure parent directories exist
            validated.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a backup if file already exists
            if validated.exists():
                backup_dir = self.guard.ensure_backup_dir(self.session_name)
                backup_path = backup_dir / f"{validated.name}.{int(time.time())}.bak"
                shutil.copy2(validated, backup_path)
            
            with open(validated, "w", encoding="utf-8") as f:
                f.write(content)
                
            return f"Success: Wrote '{path}' successfully ({len(content)} characters)."
        except Exception as e:
            return f"Error: Failed to write file '{path}'. {str(e)}"

    def patch_file(self, path: str, old_string: str, new_string: str) -> str:
        try:
            validated = self.guard.validate_path(path)
            if not validated.exists():
                return f"Error: File '{path}' not found."
            
            content = validated.read_text(encoding="utf-8", errors="ignore")
            
            # Normalize line endings to support cross-platform exact string matching
            norm_content = content.replace("\r\n", "\n")
            norm_old = old_string.replace("\r\n", "\n")
            norm_new = new_string.replace("\r\n", "\n")
            
            # Strategy 1: Exact search & replace
            if norm_old in norm_content:
                # Confirm count is exactly 1 to avoid ambiguous replacement
                occurrences = norm_content.count(norm_old)
                if occurrences > 1:
                    return f"Error: Target block has {occurrences} occurrences in the file. Please provide more surrounding lines as context to make the block unique."
                
                # Make backup before modification
                backup_dir = self.guard.ensure_backup_dir(self.session_name)
                backup_path = backup_dir / f"{validated.name}.{int(time.time())}.bak"
                shutil.copy2(validated, backup_path)
                
                new_content = norm_content.replace(norm_old, norm_new, 1)
                # Keep platform line endings if file originally had \r\n
                if "\r\n" in content and "\r\n" not in new_content:
                    new_content = new_content.replace("\n", "\r\n")
                    
                validated.write_text(new_content, encoding="utf-8")
                return f"Success: Patched '{path}' using exact match replacement."
            
            # Strategy 2: Whitespace-normalized fallback
            norm_content_collapsed = re.sub(r'\s+', ' ', norm_content)
            norm_old_collapsed = re.sub(r'\s+', ' ', norm_old)
            
            if norm_old_collapsed in norm_content_collapsed:
                # Find matching offset or report it as fuzzy
                return f"Error: Code block found but indentation or whitespace does not match. Please verify character-by-character indentation or replace the entire file block."
                
            return f"Error: Target block was not found in the file '{path}'. Please review the file content."
        except Exception as e:
            return f"Error: Failed to patch file '{path}'. {str(e)}"

    def make_directory(self, path: str) -> str:
        try:
            validated = self.guard.validate_path(path)
            if validated.exists():
                if validated.is_dir():
                    return f"Directory '{path}' already exists."
                else:
                    return f"Error: A file named '{path}' already exists at that path."
            validated.mkdir(parents=True, exist_ok=True)
            return f"Success: Created directory '{path}' successfully."
        except Exception as e:
            return f"Error: Failed to create directory '{path}'. {str(e)}"

    def delete_file(self, path: str) -> str:
        try:
            validated = self.guard.validate_path(path)
            if not validated.exists():
                return f"Error: File '{path}' does not exist."
            if validated.is_dir():
                return f"Error: '{path}' is a directory. Cannot delete directories using delete_file."
            
            backup_dir = self.guard.ensure_backup_dir(self.session_name)
            backup_path = backup_dir / f"{validated.name}.{int(time.time())}.del.bak"
            shutil.copy2(validated, backup_path)
            
            validated.unlink()
            return f"Success: Deleted file '{path}' successfully (backup saved at {backup_path.name})."
        except Exception as e:
            return f"Error: Failed to delete file '{path}'. {str(e)}"

    def delete_directory(self, path: str) -> str:
        try:
            validated = self.guard.validate_path(path)
            if not validated.exists():
                return f"Error: Directory '{path}' does not exist."
            if not validated.is_dir():
                return f"Error: '{path}' is a file. Cannot delete files using delete_directory."
            
            # Prevent deletion of workspace root
            if validated == self.guard.root:
                return "Error: Access Denied. Cannot delete the workspace root directory."
                
            backup_dir = self.guard.ensure_backup_dir(self.session_name)
            backup_path = backup_dir / f"{validated.name}.{int(time.time())}.dir_bak"
            shutil.copytree(validated, backup_path)
            
            shutil.rmtree(validated)
            return f"Success: Deleted directory '{path}' successfully (backup saved at {backup_path.name})."
        except Exception as e:
            return f"Error: Failed to delete directory '{path}'. {str(e)}"

    def list_directory(self, path: str = ".") -> str:
        try:
            validated = self.guard.validate_path(path)
            if not validated.exists():
                return f"Error: Path '{path}' not found."
            if not validated.is_dir():
                return f"Error: '{path}' is a file, use read_file instead."
                
            items = os.listdir(validated)
            dirs = []
            files = []
            for item in items:
                full = validated / item
                if full.is_dir():
                    dirs.append(item + "/")
                else:
                    files.append(item)
            
            out = f"Contents of '{path}':\n"
            out += f"Directories: {', '.join(sorted(dirs)) if dirs else 'None'}\n"
            out += f"Files: {', '.join(sorted(files)) if files else 'None'}"
            return self.truncate_result(out)
        except Exception as e:
            return f"Error: Failed to list directory. {str(e)}"

    def grep_search(self, query: str, path: str = ".") -> str:
        try:
            validated = self.guard.validate_path(path)
            if not validated.exists():
                return f"Error: Search path '{path}' does not exist."
            
            results = []
            # Walk directory recursively
            for root, dirs, filenames in os.walk(validated):
                # Filter out system and ignore folders
                dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', 'node_modules', '.venv', 'env', 'build')]
                for filename in filenames:
                    # Ignore binary/config files
                    if filename.endswith(('.png', '.jpg', '.jpeg', '.zip', '.tar', '.gz', '.pyc', '.exe', '.dll', '.so', '.hcl')):
                        continue
                    full_path = Path(root) / filename
                    # Resolve inside workspace
                    try:
                        self.guard.validate_path(str(full_path))
                    except PermissionError:
                        continue
                        
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                            for idx, line in enumerate(f, start=1):
                                if query in line or re.search(query, line, re.IGNORECASE):
                                    rel = full_path.relative_to(self.guard.root)
                                    results.append(f"{rel}:{idx}: {line.strip()}")
                                    if len(results) >= 50:
                                        break
                    except Exception:
                        pass
                if len(results) >= 50:
                    break
            
            if not results:
                return f"No matches found for '{query}' inside '{path}'."
            
            out = f"Search matches (limit 50):\n" + "\n".join(results)
            return self.truncate_result(out)
        except Exception as e:
            return f"Error: Search failed. {str(e)}"
