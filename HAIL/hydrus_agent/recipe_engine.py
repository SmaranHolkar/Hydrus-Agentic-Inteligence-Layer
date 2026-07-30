import yaml
import json
import re
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from jinja2 import Template, Environment, BaseLoader
from .session import Session
from .events import AgentEvent, EventType
from .orchestrator import SubagentTask

@dataclass
class RecipeStep:
    name: str
    action: str
    input: Optional[str] = None
    output: Optional[str] = None
    system_prompt: Optional[str] = None
    parallel: bool = False
    metacognition: bool = False
    max_retries: int = 1
    
    # Subagent config for parallel steps
    subagent_count: int = 1
    subagent_prompts: Optional[List[str]] = None

@dataclass
class Recipe:
    version: str = "1.0.0"
    title: str = "Untitled"
    description: str = ""
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[RecipeStep] = field(default_factory=list)
    metacognition: Dict[str, bool] = field(default_factory=dict)
    
    # Validation gates
    validate_steps: bool = False
    grounding_required: bool = False
    safety_review: bool = False

class RecipeEngine:
    def __init__(self, workspace: str, agent, orchestrator=None):
        self.workspace = Path(workspace)
        self.agent = agent
        self.orchestrator = orchestrator
        self.context: Dict[str, Any] = {}  # Step output registry
        self.jinja_env = Environment(loader=BaseLoader())
    
    def load_recipe(self, path: str) -> Recipe:
        """Load and validate a YAML recipe file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return self._parse_recipe(data)
    
    def _parse_recipe(self, data: Dict) -> Recipe:
        """Parse raw YAML dict into Recipe dataclass."""
        steps = []
        for step_data in data.get("steps", []):
            steps.append(RecipeStep(
                name=step_data.get("name", "unnamed"),
                action=step_data.get("action"),
                input=step_data.get("input"),
                output=step_data.get("output"),
                system_prompt=step_data.get("system_prompt"),
                parallel=step_data.get("parallel", False),
                metacognition=step_data.get("metacognition", False),
                max_retries=step_data.get("max_retries", 1),
                subagent_count=step_data.get("subagent_count", 1),
                subagent_prompts=step_data.get("subagent_prompts")
            ))
        
        meta = data.get("metacognition", {})
        
        return Recipe(
            version=data.get("version", "1.0.0"),
            title=data.get("title", "Untitled"),
            description=data.get("description", ""),
            parameters=data.get("parameters", []),
            steps=steps,
            metacognition=meta,
            validate_steps=meta.get("validate_steps", False),
            grounding_required=meta.get("grounding_required", False),
            safety_review=meta.get("safety_review", False)
        )
    
    def render_parameters(self, recipe: Recipe, user_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and render user-provided parameters."""
        rendered = {}
        
        for param_def in recipe.parameters:
            key = param_def["key"]
            required = param_def.get("required", False)
            
            if key in user_params:
                rendered[key] = user_params[key]
            elif required:
                raise ValueError(f"Missing required parameter: {key}")
            elif "default" in param_def:
                rendered[key] = param_def["default"]
        
        # Add computed parameters
        computed = {}
        for key, val in rendered.items():
            if isinstance(val, str):
                computed[f"{key}_slug"] = re.sub(r'[^\w]+', '_', val.lower()).strip('_')
        rendered.update(computed)
        
        return rendered
    
    def expand_template(self, template_str: str, params: Dict, context: Dict) -> str:
        """Expand {{variable}} and {{step.output}} references."""
        if not template_str:
            return ""
        
        # First, resolve step outputs from context
        # {{step_name.output}} or {{step_name}}
        step_pattern = r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\.output)?\s*\}\}'
        
        def step_replacer(match):
            step_name = match.group(1)
            if step_name in context:
                return str(context[step_name])
            return match.group(0)  # Keep original if not found
        
        expanded = re.sub(step_pattern, step_replacer, template_str)
        
        # Then Jinja2 render with parameters
        template = self.jinja_env.from_string(expanded)
        return template.render(**params)
    
    async def execute(self, recipe: Recipe, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a recipe step-by-step."""
        self.context = {}
        rendered_params = self.render_parameters(recipe, params)
        
        results = {
            "recipe": recipe.title,
            "steps": {},
            "final_output": None
        }
        
        for step in recipe.steps:
            print(f"[Recipe] Executing step: {step.name}")
            
            # Expand templates
            action_input = self.expand_template(step.input, rendered_params, self.context)
            
            # Grounding check before execution
            if recipe.grounding_required and step.metacognition:
                is_grounded = await self._grounding_check(action_input)
                if not is_grounded:
                    raise ValueError(f"Step {step.name} failed grounding check")
            
            # Execute
            if step.parallel and self.orchestrator and step.subagent_count > 1:
                # Parallel subagent execution
                step_result = await self._execute_parallel(step, action_input, rendered_params)
            else:
                # Sequential execution
                step_result = await self._execute_single(step, action_input)
            
            # Store in context
            self.context[step.name] = step_result
            results["steps"][step.name] = step_result
            
            # Validation gate
            if recipe.validate_steps:
                is_valid = await self._validate_step_output(step, step_result)
                if not is_valid:
                    raise ValueError(f"Step {step.name} failed validation")
        
        # Final output
        if recipe.steps:
            last_step = recipe.steps[-1]
            results["final_output"] = self.context.get(last_step.name)
        
        return results
    
    async def _execute_single(self, step: RecipeStep, action_input: str) -> str:
        """Execute a single step through the main agent."""
        prompt = f"""Execute this step: {step.name}
Action: {step.action}
Input: {action_input}

Use the appropriate tool and return the result."""
        
        if step.system_prompt:
            prompt = f"{step.system_prompt}\n\n{prompt}"
        
        session = Session(f"recipe_{step.name}", workspace=str(self.workspace))
        
        events = []
        final = None
        async for event in self.agent.run_stream(prompt, session):
            events.append(event)
            if event.type == EventType.COMPLETE:
                final = event.content
        
        # If output file specified, write it
        if step.output and final:
            output_path = self.workspace / self.expand_template(step.output, {}, self.context)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(final, encoding='utf-8')
        
        return final or "No result"
    
    async def _execute_parallel(self, step: RecipeStep, action_input: str, params: Dict) -> str:
        """Execute step with parallel subagents."""
        if not self.orchestrator:
            raise ValueError("Orchestrator required for parallel execution")
        
        # Generate subagent prompts
        if step.subagent_prompts:
            prompts = [
                self.expand_template(p, params, self.context)
                for p in step.subagent_prompts
            ]
        else:
            # Auto-generate angle prompts
            prompts = [
                f"{action_input}\nFocus on angle {i+1}."
                for i in range(step.subagent_count)
            ]
        
        tasks = [
            SubagentTask(
                id=f"{step.name}_{i}",
                prompt=prompt,
                system_override=step.system_prompt,
                max_steps=step.max_retries + 5
            )
            for i, prompt in enumerate(prompts)
        ]
        
        aggregator = f"""Synthesize the following subagent results for step '{step.name}'.
Resolve contradictions and produce a unified output."""
        
        result = await self.orchestrator.spawn_subagents(
            tasks=tasks,
            workspace=str(self.workspace),
            aggregator_prompt=aggregator
        )
        
        return result.final_answer or "Aggregation failed"
    
    async def _grounding_check(self, content: str) -> bool:
        """Check if content is grounded in available context."""
        # Placeholder: In production, query HCL for source verification
        return True
    
    async def _validate_step_output(self, step: RecipeStep, output: str) -> bool:
        """Validate step output through HCL metacognition."""
        # Placeholder: Run through HydrusOpt safety layer
        return True
