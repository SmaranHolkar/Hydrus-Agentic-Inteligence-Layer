import os
import re
from pathlib import Path
from typing import Tuple, List, Optional

class WorkspaceGuard:
    BLOCKED_COMMANDS = {
        'rm', 'rmdir', 'del', 'format', 'fdisk', 'dd', 'mkfs', 'mkfs.ext4',
        'sudo', 'su', 'chmod', 'chown', 'curl', 'wget', 'nc', 'netcat',
        'bash', 'sh', 'zsh', 'powershell', 'cmd', 'regedit', 'diskpart'
    }
    
    def __init__(self, workspace_root: str):
        self.root = Path(workspace_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
    
    def validate_path(self, path: str) -> Path:
        """Resolve path and ensure it's within workspace."""
        target = Path(path)
        
        # Handle relative paths
        if not target.is_absolute():
            target = self.root / target
        
        target = target.resolve()
        
        # Check traversal
        try:
            target.relative_to(self.root)
        except ValueError:
            raise PermissionError(
                f"Path '{path}' resolves to '{target}' which is outside workspace '{self.root}'"
            )
        
        # Check symlink escape
        if target.is_symlink():
            real = target.readlink()
            if not real.is_absolute():
                real = self.root / real
            real = real.resolve()
            try:
                real.relative_to(self.root)
            except ValueError:
                raise PermissionError(f"Symlink '{path}' escapes workspace")
        
        return target
    
    def validate_command(self, command: str) -> Tuple[bool, str, str]:
        """
        Validate command.
        Returns: (is_allowed, danger_level, reason)
        danger_level: low | medium | high
        """
        if not command or not command.strip():
            return False, "high", "Empty command"
        
        # Extract base command (handle pipes, redirects, etc.)
        cmd_clean = command.strip().split('|')[0].split('>')[0].split('&')[0]
        parts = cmd_clean.strip().split()
        if not parts:
            return False, "high", "Empty command after filtering"
        base = parts[0].lower().strip('"\'')
        
        # Check blocked list
        if base in self.BLOCKED_COMMANDS:
            return False, "high", f"Command '{base}' is blocked"
        
        # Danger level scoring
        danger = "low"
        lowered = command.lower()
        
        high_indicators = ['rm', 'rmdir', 'del', 'format', '>', '>>', 'chmod', 'chown', 'curl', 'wget']
        medium_indicators = ['git', 'npm', 'pip', 'yarn', 'pnpm', 'docker', 'pytest', 'python']
        
        if any(ind in lowered for ind in high_indicators):
            danger = "high"
        elif any(ind in lowered for ind in medium_indicators):
            danger = "medium"
        
        # Check for absolute path references or home dir references outside workspace
        abs_paths = re.findall(r'[/~][^\s|&>]+', command)
        for p in abs_paths:
            if p.startswith('~'):
                return False, "high", "Home directory references not allowed"
            # On windows, check for drive letters (e.g. C:\)
            if re.match(r'^[a-zA-Z]:\\', p) or p.startswith('\\'):
                # Check if it resolves within workspace
                try:
                    Path(p).resolve().relative_to(self.root)
                except ValueError:
                    return False, "high", f"Access to path '{p}' outside workspace is blocked"
        
        return True, danger, "Command allowed"
    
    def ensure_backup_dir(self, session_name: str) -> Path:
        backup = self.root / "data" / "backups" / session_name
        backup.mkdir(parents=True, exist_ok=True)
        return backup


class PromptInjectionDetector:
    """Detects common prompt injection and jailbreak patterns."""
    
    PATTERNS = [
        # Ignore previous instructions
        r'ignore\s+(?:all\s+)?(?:previous|prior|earlier)\s+(?:instructions|prompts|commands)',
        # System prompt leaks
        r'(?:repeat|print|echo|output)\s+(?:your|the)\s+(?:system|initial|original)\s+(?:prompt|instructions)',
        # DAN / jailbreak prefixes
        r'\bDAN\b|Do Anything Now|jailbreak|developer mode',
        # Role override attempts
        r'you are now\s+\w+|from now on you are|act as\s+\w+',
        # Delimiter confusion
        r'<\s*/\s*system\s*>|<\s*/\s*instruction\s*>|###\s*SYSTEM',
        # Indirect injection via data
        r'\{\{.*?\}\}|<%.*?%>|`\{.*?\}`',
        # Multi-language obfuscation
        r'(?:system|ignore|override|bypass).*?(?:instruction|prompt|restriction)',
    ]
    
    DANGEROUS_COMMANDS = [
        r'rm\s+-rf\s+[/\\]',
        r'dd\s+if=/dev/zero',
        r'mkfs',
        r'format\s+[c-z]:',
        r'del\s+/[fq]\s+.*\\',
        r'powershell\s+-enc',
        r'base64\s+-d\s*\|',
        r'curl.*\|\s*bash',
        r'wget.*\|\s*sh',
    ]
    
    def __init__(self):
        self.injection_regex = re.compile(
            '|'.join(f'({p})' for p in self.PATTERNS),
            re.IGNORECASE | re.DOTALL
        )
        self.command_regex = re.compile(
            '|'.join(f'({p})' for p in self.DANGEROUS_COMMANDS),
            re.IGNORECASE
        )
    
    def scan_input(self, text: str) -> Tuple[bool, List[str]]:
        """
        Scan user input for injection attempts.
        Returns: (is_safe, list_of_detected_patterns)
        """
        matches = []
        
        for pattern in self.PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                matches.append(f"Injection pattern: '{match.group()}' at position {match.start()}")
        
        for pattern in self.DANGEROUS_COMMANDS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append(f"Dangerous command: '{match.group()}' at position {match.start()}")
        
        return len(matches) == 0, matches
    
    def sanitize_for_display(self, text: str) -> str:
        """Sanitize text for safe display (prevent XSS in dashboard)."""
        # Basic HTML escaping
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        return text


class SecurityManager:
    """Combined security manager for HydrusAgent."""
    
    def __init__(self, workspace: str):
        self.workspace = workspace
        self.guard = WorkspaceGuard(workspace)
        self.detector = PromptInjectionDetector()
    
    def validate_user_input(self, prompt: str) -> Tuple[bool, str]:
        """Pre-process user input before sending to agent."""
        is_safe, violations = self.detector.scan_input(prompt)
        if not is_safe:
            return False, f"Security alert: {len(violations)} violations detected:\n" + "\n".join(violations)
        return True, prompt
    
    def validate_agent_output(self, output: str, context: str = "") -> Tuple[bool, str]:
        """Post-process agent output before returning to user."""
        # Check for leaked system prompts
        if "system prompt" in output.lower() or "instructions" in output.lower():
            if len(output) > 500:  # Likely a leak attempt
                return False, "Output blocked: potential system prompt leak detected"
        
        # Check for dangerous commands in output
        is_safe, violations = self.detector.scan_input(output)
        if not is_safe:
            return False, f"Output blocked: dangerous content detected:\n" + "\n".join(violations)
        
        return True, output


class SecurityIngestionGuard:
    """
    Active Ingestion Firewall for external MCP payloads.
    Protects Cognitive Memory Gateway (CMG) from secrets leak, PII, and indirect prompt injection attacks.
    """
    
    SECRET_PATTERNS = [
        (r'AKIA[0-9A-Z]{16}', 'AWS_ACCESS_KEY'),
        (r'ghp_[a-zA-Z0-9]{36}', 'GITHUB_PAT'),
        (r'sk-[a-zA-Z0-9]{32,}', 'API_SECRET_KEY'),
        (r'-----BEGIN\s+(?:RSA|OPENSSH|EC|PRIVATE)\s+KEY-----[\s\S]*?-----END', 'PRIVATE_KEY'),
        (r'(?:api[_-]?key|secret[_-]?token|bearer[_-]?token)\s*[:=]\s*["\']?[a-zA-Z0-9_\-\.]{16,}["\']?', 'GENERIC_TOKEN')
    ]
    
    PII_PATTERNS = [
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL'),
        (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN'),
        (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', 'CREDIT_CARD'),
        (r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b', 'IPV4')
    ]
    
    INDIRECT_INJECTION_PATTERNS = [
        r'ignore\s+(?:all\s+)?(?:previous|prior|earlier)\s+(?:instructions|prompts|commands)',
        r'system\s+override',
        r'you\s+are\s+now\s+a\s+different\s+model',
        r'act\s+as\s+DAN',
        r'drop\s+database',
        r'delete\s+all\s+files',
        r'exec\s*\(\s*["\'].*["\']\s*\)',
        r'<\s*system\s*>',
        r'\[\s*SYSTEM\s*INSTRUCTION\s*\]'
    ]
    
    def __init__(self):
        self.injection_regexes = [re.compile(p, re.IGNORECASE) for p in self.INDIRECT_INJECTION_PATTERNS]
    
    def inspect_and_sanitize(self, payload: str, source_type: str = "external_mcp") -> dict:
        """
        Scan payload for RAG indirect prompt injection, PII, and secrets.
        Returns sanitization result dict.
        """
        if not payload:
            return {
                "is_safe": True,
                "sanitized_payload": "",
                "quarantined": False,
                "quarantine_reason": None,
                "security_tag": "public-api",
                "redacted_count": 0
            }
            
        quarantine_reasons = []
        
        # 1. Indirect Prompt Injection Firewall Scan
        for regex in self.injection_regexes:
            match = regex.search(payload)
            if match:
                quarantine_reasons.append(f"Indirect prompt injection pattern detected: '{match.group()}'")
                
        if quarantine_reasons:
            return {
                "is_safe": False,
                "sanitized_payload": f"[QUARANTINED CONTENT - {'; '.join(quarantine_reasons)}]",
                "quarantined": True,
                "quarantine_reason": "; ".join(quarantine_reasons),
                "security_tag": "system-critical",
                "redacted_count": 0
            }
            
        # 2. PII & Secret Redaction
        sanitized = payload
        redacted_count = 0
        
        for pattern, label in self.SECRET_PATTERNS:
            matches = list(re.finditer(pattern, sanitized, re.IGNORECASE))
            if matches:
                redacted_count += len(matches)
                sanitized = re.sub(pattern, f"[REDACTED_SECRET:{label}]", sanitized, flags=re.IGNORECASE)
                
        for pattern, label in self.PII_PATTERNS:
            matches = list(re.finditer(pattern, sanitized))
            if matches:
                redacted_count += len(matches)
                sanitized = re.sub(pattern, f"[REDACTED_PII:{label}]", sanitized)
                
        # 3. Security Tagging
        if source_type in ("postgres", "database", "crm"):
            security_tag = "authenticated-db"
        elif source_type in ("filesystem", "local_doc"):
            security_tag = "user-private"
        else:
            security_tag = "public-api"
            
        return {
            "is_safe": True,
            "sanitized_payload": sanitized,
            "quarantined": False,
            "quarantine_reason": None,
            "security_tag": security_tag,
            "redacted_count": redacted_count
        }

