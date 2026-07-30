import click
import asyncio
import os
import sys
import json
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text

# Ensure parent directory is in path if executed directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydrus_agent.agent import HydrusAgent
from hydrus_agent.llm_bridge import HydrusOptAdapter
from hydrus_agent.permissions import CLIPermissionGate
from hydrus_agent.session import Session
from hydrus_agent.events import EventType

console = Console()

async def run_chat_loop(model_name: str, workspace_path: str):
    console.print(Panel(
        f"[bold cyan]Welcome to HydrusAgent CLI Console[/bold cyan]\n"
        f"Model: {model_name}\n"
        f"Workspace: {workspace_path}\n"
        f"Type 'exit' or 'quit' to end the session.",
        border_style="cyan"
    ))
    
    # Initialize adapter, agent, and session
    adapter = HydrusOptAdapter(model_name=model_name)
    agent = HydrusAgent(
        workspace=workspace_path,
        model_adapter=adapter,
        permission_gate=CLIPermissionGate()
    )
    session = Session(name="cli_session", workspace=workspace_path)
    
    while True:
        try:
            user_input = console.input("[bold green]Operator > [/bold green]").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                console.print("[bold yellow]Ending session. Goodbye![/bold yellow]")
                break
                
            console.print("\n[bold cyan]Assistant is thinking...[/bold cyan]")
            
            # Run stream
            async for event in agent.run_stream(user_input, session):
                if event.type == EventType.THOUGHT:
                    console.print(f"[bold magenta]🧠 Thought:[/bold magenta] {event.content}")
                elif event.type == EventType.ACTION:
                    tool_name = event.metadata.get("tool", "Unknown")
                    tool_args = event.metadata.get("args", {})
                    console.print(f"[bold yellow]⚙️ Action:[/bold yellow] Call tool '{tool_name}' with args {tool_args}")
                elif event.type == EventType.OBSERVATION:
                    # Collapsible or neat display of observation
                    trunc = event.content
                    if len(trunc) > 500:
                        trunc = trunc[:500] + "\n... [truncated]"
                    console.print(Panel(trunc, title="📝 Observation", border_style="dim"))
                elif event.type == EventType.SAFETY_REVIEW:
                    console.print(f"[bold blue]ℹ️ HCL Grounding/Status:[/bold blue] {event.content}")
                elif event.type == EventType.ERROR:
                    console.print(f"[bold red]❌ Error:[/bold red] {event.content}")
                elif event.type == EventType.COMPLETE:
                    console.print("\n[bold green]🏁 Final Answer:[/bold green]")
                    console.print(Markdown(event.content))
                    console.print()
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Session interrupted. Goodbye![/bold yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]System Exception in ReAct loop: {str(e)}[/bold red]")

@click.group()
def main():
    """HydrusAgent Command Line Developer Hub."""
    pass

@click.command(name="chat")
@click.option("--model", default="microsoft/Phi-3.5-mini-instruct", help="The name of the local model to run.")
@click.option("--workspace", default=".", help="Workspace path to use as project root.")
def chat(model, workspace):
    """Start an interactive chat session with HydrusAgent."""
    workspace_abs = os.path.abspath(workspace)
    asyncio.run(run_chat_loop(model, workspace_abs))

main.add_command(chat)

@click.command(name="run")
@click.argument("recipe_path")
@click.option("--param", "-p", multiple=True, help="Parameters as key=value")
@click.option("--model", default="microsoft/Phi-3.5-mini-instruct", help="The name of the local model to run.")
def run_recipe(recipe_path, param, model):
    """Execute a YAML recipe."""
    from hydrus_agent.recipe_engine import RecipeEngine
    from hydrus_agent.orchestrator import AsyncSubagentOrchestrator
    
    # Parse params
    params = {}
    for p in param:
        if "=" in p:
            k, v = p.split("=", 1)
            params[k] = v
            
    # Load model and adapters
    adapter = HydrusOptAdapter(model_name=model)
    orchestrator = AsyncSubagentOrchestrator(main_adapter=adapter)
    
    agent = HydrusAgent(
        workspace=".",
        model_adapter=adapter,
        permission_gate=CLIPermissionGate()
    )
    
    engine = RecipeEngine(".", agent=agent, orchestrator=orchestrator)
    recipe = engine.load_recipe(recipe_path)
    
    console.print(Panel(f"[bold]📜 Running Recipe: {recipe.title}[/bold]", border_style="blue"))
    
    result = asyncio.run(engine.execute(recipe, params))
    
    console.print(Panel(f"[bold green]✅ Complete[/bold green]", border_style="green"))
    console.print_json(json.dumps(result, indent=2))

main.add_command(run_recipe)

if __name__ == "__main__":
    main()
