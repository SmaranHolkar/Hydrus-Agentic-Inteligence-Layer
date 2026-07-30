import asyncio
import json
from typing import List, Dict, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from .agent import HydrusAgent
from .llm_bridge import HydrusOptAdapter
from .session import Session
from .events import AgentEvent, EventType

@dataclass
class SubagentTask:
    id: str
    prompt: str
    system_override: Optional[str] = None
    max_steps: int = 10
    priority: int = 0  # Lower = higher priority

@dataclass
class SubagentResult:
    task_id: str
    events: List[AgentEvent]
    final_answer: Optional[str] = None
    error: Optional[str] = None

class AsyncSubagentOrchestrator:
    """
    Manages subagent execution with VRAM-aware scheduling.
    - Single GPU: Sequential queue (one LLM inference at a time)
    - Multi-GPU: Parallel across devices
    - Lightweight mode: Smaller model for subagents
    """
    
    def __init__(self, 
                 main_adapter: HydrusOptAdapter,
                 max_concurrent: int = 1,  # Default to 1 for single GPU safety
                 lightweight_adapter: Optional[HydrusOptAdapter] = None):
        self.main_adapter = main_adapter
        self.lightweight_adapter = lightweight_adapter or main_adapter
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.results: Dict[str, SubagentResult] = {}
    
    async def spawn_subagents(self, 
                              tasks: List[SubagentTask],
                              workspace: str,
                              aggregator_prompt: Optional[str] = None) -> SubagentResult:
        """
        Spawn multiple subagents and aggregate their results.
        
        If max_concurrent=1, runs sequentially to prevent OOM.
        If max_concurrent>1, runs in parallel across GPU streams.
        """
        # Queue all tasks
        for task in tasks:
            await self.task_queue.put(task)
        
        # Execute with semaphore-controlled concurrency
        workers = [
            asyncio.create_task(self._worker_loop(workspace, i))
            for i in range(min(self.max_concurrent, len(tasks)))
        ]
        
        # Wait for all tasks to complete
        await self.task_queue.join()
        
        # Cancel workers
        for w in workers:
            w.cancel()
        
        # Aggregate results
        return await self._aggregate_results(tasks, aggregator_prompt)
    
    async def _worker_loop(self, workspace: str, worker_id: int):
        """Worker that processes tasks from the queue."""
        while True:
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            
            try:
                result = await self._execute_single(task, workspace)
                self.results[task.id] = result
            except Exception as e:
                self.results[task.id] = SubagentResult(
                    task_id=task.id,
                    events=[],
                    error=str(e)
                )
            finally:
                self.task_queue.task_done()
    
    async def _execute_single(self, task: SubagentTask, workspace: str) -> SubagentResult:
        """Execute one subagent task."""
        async with self.semaphore:
            # Use lightweight adapter for subagents if available
            adapter = self.lightweight_adapter if task.priority > 0 else self.main_adapter
            
            agent = HydrusAgent(
                workspace=workspace,
                model_adapter=adapter,
                max_steps=task.max_steps
            )
            
            session = Session(f"subagent_{task.id}", workspace=workspace)
            
            # Override system prompt if provided
            if task.system_override:
                # Inject into session context
                session.add_message("system", task.system_override)
            
            events = []
            final_answer = None
            
            async for event in agent.run_stream(task.prompt, session):
                events.append(event)
                if event.type == EventType.COMPLETE:
                    final_answer = event.content
                # Small yield to prevent blocking event loop
                await asyncio.sleep(0)
            
            return SubagentResult(
                task_id=task.id,
                events=events,
                final_answer=final_answer
            )
    
    async def _aggregate_results(self, 
                                  tasks: List[SubagentTask],
                                  aggregator_prompt: Optional[str] = None) -> SubagentResult:
        """Merge subagent outputs into a single coherent result."""
        # Collect all answers
        answers = []
        for task in tasks:
            result = self.results.get(task.id)
            if result and result.final_answer:
                answers.append(f"[Subagent {task.id}]\n{result.final_answer}")
            elif result and result.error:
                answers.append(f"[Subagent {task.id}]\nError: {result.error}")
        
        combined = "\n\n---\n\n".join(answers)
        
        if aggregator_prompt:
            # Run aggregation through main adapter
            prompt = f"""{aggregator_prompt}

SUBAGENT RESULTS:
{combined}

Synthesize the above results into a unified, coherent answer. Resolve any contradictions."""
            
            # Use main agent for final synthesis
            session = Session("aggregator", workspace=".")
            agent = HydrusAgent(workspace=".", model_adapter=self.main_adapter)
            
            events = []
            final = None
            async for event in agent.run_stream(prompt, session):
                events.append(event)
                if event.type == EventType.COMPLETE:
                    final = event.content
            
            return SubagentResult(
                task_id="aggregator",
                events=events,
                final_answer=final
            )
        
        return SubagentResult(
            task_id="raw_aggregate",
            events=[],
            final_answer=combined
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Return current queue status for dashboard monitoring."""
        return {
            "queue_size": self.task_queue.qsize(),
            "max_concurrent": self.max_concurrent,
            "completed_tasks": len(self.results),
            "active_workers": self.max_concurrent - self.semaphore._value
        }
