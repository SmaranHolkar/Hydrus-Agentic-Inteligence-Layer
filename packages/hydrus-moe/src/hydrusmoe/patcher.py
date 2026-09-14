"""
Plug-and-Play HuggingFace & PyTorch Model Patcher for HydrusMoE.

Allows developers to accelerate and stream MoE models with a 2-line wrapper:
    import hydrusmoe
    engine = hydrusmoe.patch(model, vram_budget_mb=4096, enable_prefetch=True)
"""

from typing import Any, Optional, Dict, Union
from pathlib import Path
import torch
import torch.nn as nn

from .config import HydrusMoEConfig
from .engine import HydrusMoEEngine


class HydrusMoEWrapper(nn.Module):
    """
    Drop-in wrapper that intercepts forward and generate passes on MoE models,
    driving tiered storage offloading and context-driven predictive prefetching.
    """

    def __init__(self, model: nn.Module, engine: HydrusMoEEngine):
        super().__init__()
        self.model = model
        self.engine = engine

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # Pre-pass: Trigger context prefetching if token sequences or prompt strings are present
        prompt_str = ""
        if "input_ids" in kwargs and isinstance(kwargs["input_ids"], torch.Tensor):
            tokens = kwargs["input_ids"].tolist()
            if tokens and isinstance(tokens[0], list):
                prompt_str = " ".join(str(t) for t in tokens[0][:64])
            elif tokens:
                prompt_str = " ".join(str(t) for t in tokens[:64])
        elif len(args) > 0 and isinstance(args[0], torch.Tensor):
            tokens = args[0].tolist()
            if tokens and isinstance(tokens[0], list):
                prompt_str = " ".join(str(t) for t in tokens[0][:64])
            elif tokens:
                prompt_str = " ".join(str(t) for t in tokens[:64])
        elif "prompt" in kwargs and isinstance(kwargs["prompt"], str):
            prompt_str = kwargs["prompt"]

        if prompt_str and hasattr(self.engine, "prefetcher"):
            try:
                predicted = self.engine.prefetcher.predict(prompt_str)
                if predicted and hasattr(self.engine, "_submit_prefetch_async"):
                    self.engine._submit_prefetch_async(set(predicted))
            except Exception:
                pass

        # Execute model forward pass
        return self.model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self.model, "generate"):
            return self.model.generate(*args, **kwargs)
        return self.forward(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def patch(
    model: nn.Module,
    vram_budget_gb: float = 4.0,
    ram_budget_gb: float = 8.0,
    vram_budget_mb: Optional[int] = None,
    ram_budget_mb: Optional[int] = None,
    ssd_cache_dir: Union[str, Path] = "./.hydrusmoe_cache",
    enable_prefetch: bool = True,
    config: Optional[HydrusMoEConfig] = None,
) -> HydrusMoEWrapper:
    """
    Patches a PyTorch / HuggingFace MoE model with HydrusMoE 4-tier streaming engine.
    
    Args:
        model: HuggingFace or PyTorch model
        vram_budget_gb: GPU VRAM limit in Gigabytes (default: 4.0 GB)
        ram_budget_gb: Host RAM cache limit in Gigabytes (default: 8.0 GB)
        vram_budget_mb: (Optional) GPU VRAM limit in Megabytes
        ram_budget_mb: (Optional) Host RAM cache limit in Megabytes
        ssd_cache_dir: Local SSD cache path for encrypted expert shards
        enable_prefetch: Whether to enable speculative predictive prefetching
        config: Optional custom HydrusMoEConfig
        
    Returns:
        HydrusMoEWrapper wrapping the model with active streaming engine
    """
    if config is None:
        vram_gb = (vram_budget_mb / 1024.0) if vram_budget_mb is not None else vram_budget_gb
        ram_gb = (ram_budget_mb / 1024.0) if ram_budget_mb is not None else ram_budget_gb
        config = HydrusMoEConfig(
            vram_budget_gb=vram_gb,
            ram_budget_gb=ram_gb,
            ssd_cache_dir=Path(ssd_cache_dir),
            enable_hail_prefetch=enable_prefetch,
        )

    engine = HydrusMoEEngine(config=config)
    return HydrusMoEWrapper(model=model, engine=engine)

