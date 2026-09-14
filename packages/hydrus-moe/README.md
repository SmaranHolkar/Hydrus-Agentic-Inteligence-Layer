# ⚡ HydrusMoE: 4-Tier Streamable Mixture-of-Experts Engine

> **Run 14B–35B+ MoE Models on 4–8GB Consumer GPUs with Hardware-Bound Zero-Trust Security & Speculative Prefetching.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)

**HydrusMoE** is a drop-in acceleration SDK that brings multi-tiered memory streaming (VRAM $\rightarrow$ Host RAM $\rightarrow$ SSD Vault $\rightarrow$ Cloud CDN) and context-driven predictive prefetching ($P > 0.75$) to consumer hardware.

---

## 🚀 Quickstart (2-Line Model Patch)

```python
import torch
from transformers import AutoModelForCausalLM
import hydrusmoe

# 1. Load any MoE model on CPU / lightweight device map
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B", device_map="cpu")

# 2. Patch with HydrusMoE streaming & prefetching
patched_model = hydrusmoe.patch(
    model,
    vram_budget_mb=4096,     # Pinned in GPU memory (4GB VRAM)
    ram_budget_mb=12288,     # Host mlock buffer (12GB RAM)
    enable_prefetch=True     # Context-driven speculative prefetching
)

# Standard inference pipeline remains identical
output = patched_model.generate(**inputs)
```

---

## 🛡️ Zero-Trust Security Built-In

* **Hardware-Bound AES-256-GCM**: Weight shards are encrypted locally with hardware-derived keys (`HKDF-SHA256(seed || hardware_uuid)`). Copied files cannot be decrypted on other machines.
* **Access-Pattern Obfuscation**: Real expert requests are padded with decoy dummy experts (up to batch size 8) to prevent side-channel memory-profiling attacks.
* **Cryptographic Verification**: Ed25519 signature validation and Merkle tree root hash checking on shard loads.
