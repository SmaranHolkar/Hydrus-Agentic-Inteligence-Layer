import re
import torch
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self, *args, **kwargs: self.get_seq_length()
except ImportError:
    pass

class StopOnAction(StoppingCriteria):
    """Stop generation when </action> is emitted."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stop_seq = "</action>"
        if hasattr(self.tokenizer, "encode"):
            self.stop_tokens = self.tokenizer.encode(self.stop_seq, add_special_tokens=False)
        else:
            self.stop_tokens = []
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if not self.stop_tokens:
            return False
        # Check if the last tokens match the stop sequence
        if len(input_ids[0]) >= len(self.stop_tokens):
            last = input_ids[0][-len(self.stop_tokens):].tolist()
            if last == self.stop_tokens:
                return True
        return False

class HydrusOptAdapter:
    def __init__(self, model_name: str = "microsoft/Phi-3.5-mini-instruct", device: str = "auto", model=None, tokenizer=None):
        cache_dir = r"D:\HydrusOPT\models"
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            print("[LLMBridge] Using existing in-memory model and tokenizer.")
        else:
            print(f"[LLMBridge] Loading model {model_name} from cache {cache_dir}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            if not torch.cuda.is_available():
                self.model = self.model.to("cpu")
        
        if hasattr(self.model, "parameters"):
            try:
                self.device = next(self.model.parameters()).device
            except StopIteration:
                self.device = "cpu"
        else:
            self.device = "cpu"
        self.stop_criteria = StoppingCriteriaList([StopOnAction(self.tokenizer)])
    
    def format_tools(self, tools: List[Dict]) -> str:
        """Format tool definitions for ReAct prompt."""
        lines = []
        for tool in tools:
            params = ", ".join([
                f'{k}: {v.get("type", "string")}' 
                for k, v in tool.get("parameters", {}).get("properties", {}).items()
            ])
            lines.append(f"- {tool['name']}: {tool['description']} | Args: {{{params}}}")
        return "\n".join(lines)
    
    def build_react_prompt(self, user_message: str, tools: List[Dict], history: List[Dict], workspace: Optional[str] = None, context: Optional[str] = None) -> str:
        tool_desc = self.format_tools(tools)
        ws_info = f"Current Workspace Root: {workspace}\n" if workspace else ""
        
        system = f"""You are HydrusAgent, a precise developer assistant.
You solve tasks step-by-step using available tools.

Your capabilities include:
- Native File Operations: read_file (range-restricted), write_file, make_directory, delete_file, delete_directory, list_directory, grep_search, and patch_file (indentation/whitespace-normalized search and replace).
- Secure Shell Commands: execute commands via run_command (capped at 30s timeouts, safe filters).
- Native Git Operations: check git_status, git_diff, git_add, git_commit, git_log, git_branch, and git_init within the workspace.
- Deep Web Research: query the web (web_search) and parse page content to markdown (fetch_webpage).
- Recurrent YAML Workflows: parse and run multi-step Recipes containing sequential pipelines.
- Parallel Subagent Orchestration: execute parallel subagents with queue-based VRAM safety controls to prevent GPU memory OOMs.
- HCL Memory Integration: retrieve past facts/grounding contexts from HCL's Stratified Memory Lattice (SML) and commit observations as episodic memories.
- Safety & Security: block sandbox escapes via WorkspaceGuard path validation, reject destructive commands, and filter prompt injection jailbreaks.

{ws_info}
AVAILABLE TOOLS:
{tool_desc}

RULES:
1. ALWAYS think step-by-step inside <thought> tags.
2. To use a tool, write EXACTLY: <action>ToolName|{{"arg1": "value1"}}</action>
3. After seeing a tool result, continue inside <thought> tags.
4. When the task is complete, write: <action>FinalAnswer|{{"answer": "..."}}</action>
5. NEVER write code or explanations outside <thought> or <action> tags.
6. NEVER hallucinate tool results. Wait for the actual result.
7. Always construct file paths as relative paths (e.g. 'hello/hello.txt' or './hello.txt') relative to the current workspace root. Do NOT use absolute Unix paths like '/home/user/...' or '/tmp/...'. All operations must remain inside the workspace root."""

        if context and context.strip():
            system += f"\n\nMemory Grounding Context from past sessions:\n{context.strip()}"

        messages = [{"role": "system", "content": system}]
        
        for h in history[-8:]:
            content = h["content"]
            role = h["role"]
            if role == "user":
                messages.append({"role": "user", "content": content})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": content})
            elif role == "observation":
                messages.append({"role": "assistant", "content": f"Observation: {content}"})
        
        messages.append({"role": "user", "content": user_message})

        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None) is not None:
            try:
                templated = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                return templated + "<thought>"
            except Exception as e:
                print(f"[LLMBridge] apply_chat_template failed: {e}")

        history_str = ""
        for h in history[-8:]:
            if h["role"] == "user":
                history_str += f"User: {h['content']}\n"
            elif h["role"] == "assistant":
                history_str += f"Assistant: {h['content']}\n"
            elif h["role"] == "observation":
                history_str += f"Observation: {h['content']}\n"
        
        prompt = f"""{system}

{history_str}
User: {user_message}

<thought>"""
        return prompt
    
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Check system VRAM limits or device
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                stopping_criteria=self.stop_criteria,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return generated.strip()
    
    def parse_react_output(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[dict]]:
        """
        Parse ReAct output into (thought, action_name, action_args).
        Returns (None, None, None) if no action found.
        """
        # Extract thought (handles both repeated <thought> tag or direct continuation after <thought> template suffix)
        thought_match = re.search(r'<thought>(.*?)(?:</thought>|<action>|$)', text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        else:
            thought_fallback = re.search(r'^(.*?)(?:</thought>|<action>|$)', text, re.DOTALL)
            thought = thought_fallback.group(1).strip() if thought_fallback else ""
        
        # Extract action
        action_match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
        action_str = None
        if action_match:
            action_str = action_match.group(1).strip()
        else:
            action_match_open = re.search(r'<action>(.*)$', text, re.DOTALL)
            if action_match_open:
                action_str = action_match_open.group(1).strip()
            else:
                # Try fallback matching for "Action: ToolName|{...}" or plain Action prefix
                action_fallback = re.search(r'(?:Action|action):\s*(.*)$', text, re.DOTALL | re.IGNORECASE)
                if action_fallback:
                    action_str = action_fallback.group(1).strip()
                else:
                    # Look for plain "ToolName|{...}"
                    plain_fallback = re.search(r'([a-zA-Z0-9_]+\s*\|.*)$', text, re.DOTALL)
                    if plain_fallback:
                        action_str = plain_fallback.group(1).strip()
        
        if not action_str:
            return thought, None, None
            
        # Parse ToolName|{"args"}
        if '|' not in action_str:
            # If no pipe, but contains a JSON block or arguments (frequent in raw text fallbacks)
            # e.g., "FinalAnswer' with args {"answer": "..."}"
            json_match = re.search(r'(\{.*\})', action_str, re.DOTALL)
            if json_match:
                args_str = json_match.group(1).strip()
                tool_name_match = re.match(r'^([a-zA-Z0-9_]+)', action_str)
                tool_name = tool_name_match.group(1).strip() if tool_name_match else action_str.strip()
            else:
                return thought, action_str, {}
        else:
            tool_name, args_str = action_str.split('|', 1)
            tool_name = tool_name.strip()
        
        # Strip trailing </action> if it leaked in fallback
        if args_str.endswith("</action>"):
            args_str = args_str[:-9].strip()
            
        try:
            import json
            # Attempt to parse after trimming excess trailing braces (frequent with small models)
            cleaned = args_str.strip()
            while cleaned.endswith('}') and cleaned.count('}') > cleaned.count('{'):
                cleaned = cleaned[:-1].strip()
            args = json.loads(cleaned)
        except Exception:
            # Fallback regex parsing for key arguments to recover from formatting errors
            args = {}
            query_match = re.search(r'"query"\s*:\s*"([^"]+)"', args_str)
            url_match = re.search(r'"url"\s*:\s*"([^"]+)"', args_str)
            path_match = re.search(r'"path"\s*:\s*"([^"]+)"', args_str)
            command_match = re.search(r'"command"\s*:\s*"([^"]+)"', args_str)
            answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', args_str)
            
            if query_match:
                args["query"] = query_match.group(1)
            if url_match:
                args["url"] = url_match.group(1)
            if path_match:
                args["path"] = path_match.group(1)
            if command_match:
                args["command"] = command_match.group(1)
            if answer_match:
                args["answer"] = answer_match.group(1)
                
            if not args:
                cleaned_args = args_str.strip().strip('"\'')
                if cleaned_args.startswith('{'):
                    args = {"raw": args_str.strip()}
                else:
                    if tool_name in ["read_file", "list_directory", "delete_file"]:
                        args = {"path": cleaned_args}
                    elif tool_name == "run_command":
                        args = {"command": cleaned_args}
                    elif tool_name == "web_search":
                        args = {"query": cleaned_args}
                    else:
                        args = {"raw": cleaned_args}
        
        return thought, tool_name, args
