"""
test_bni.py — Belief Navigation Index (BNI) + Belief Layer Interventions
=========================================================================
Two-phase design:

  Phase 1 — Probe (read-only, single forward pass):
    Maps the model's belief trajectory across sampled layers using three graph
    traversal algorithms.  Identifies two critical layer targets:

      A. "Commitment layer"  — first sampled layer where the final answer token
                               enters argmax position.
      B. "Turbulence layer"  — sampled layer with the worst argmax flip
                               (highest local JSD between adjacent layers).

  Phase 2 — Intervene (two surgical forward hooks, no training):
    Hook A — Belief Anchor Hook (BAH):
      Installed on the commitment layer.  Amplifies the hidden-state component
      that points toward the top-token's unembedding direction by factor alpha.
      Effect: makes belief "stickier" after it first commits — later layers are
      less likely to backtrack.

    Hook B — Belief Turbulence Damper (BTD):
      Installed on the turbulence layer.  Blends the layer's output hidden state
      with a running mean of the previous layer's output using weight beta.
      Effect: smooths the spike that causes the worst argmax flip.

  After intervention the prompt is re-generated and BNI is recomputed so you
  can see the delta.

Graph algorithms (used in both phases):
  1. Belief BFS      — answer-token emergence depth
  2. Belief DFS      — argmax commitment stability
  3. Dijkstra        — confidence-weighted minimum-cost belief path (skip edges)

  BNI = 0.25·BFS + 0.35·DFS + 0.40·Dijkstra  ∈ [0, 1]

GPU-safe:
  - 4-bit NF4 quantization by default
  - Only a sparse set of layers is sampled (default 8)
  - lm_head projection on CPU (no VRAM spike)
  - Hooks operate in-place on residual stream (no extra matrices)

Usage:
  cd D:\\HydrusOPT
  python tests/test_bni.py
  python tests/test_bni.py --model Qwen/Qwen2.5-3B-Instruct --n-layers 6
  python tests/test_bni.py --alpha 1.2 --beta 0.25
  python tests/test_bni.py --skip-quant --prompts "What is 2+2?" "Paris is the capital of Germany."
"""

import sys
import os
import re
import warnings
import heapq
import argparse
from typing import List, Tuple, Set, Dict, Any, Optional

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore", message=".*_check_is_size.*",           category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*",  category=UserWarning)
warnings.filterwarnings("ignore", message=".*huggingface_hub.*symlinks.*", category=UserWarning)

# Use cached weights without hitting HF Hub — avoids rate-limited slow downloads
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Test prompts ───────────────────────────────────────────────────────────────
DEFAULT_PROMPTS: List[str] = [
    # Confident factual — expect high BNI
    "What is the capital of France?",
    "What is 2 + 2? Answer with one integer only.",
    "How many moons does Jupiter have? Answer with one integer only.",
    # Factually false — model may waver → lower BNI
    "The capital of France is London.",
    # Ambiguous / paradox — expect low BNI (unstable traversal)
    "Is this statement false: this statement is false?",
    "What is the color of silence?",
    "Can God create a rock so heavy that even God cannot lift it?",
]

# ── Layer sampling ─────────────────────────────────────────────────────────────

def _sample_layer_indices(num_layers: int, n_samples: int = 8) -> List[int]:
    """
    Return n_samples evenly-spaced layer indices across the useful range.

    Starts from layer 1 (not 0) — layer 0 is embedding output which produces
    vocabulary noise before any transformer block has run, making argmax
    meaningless and polluting BFS/DFS scores.
    Always includes the final layer.

    Uniform spacing is intentional: the variation between early (noisy) and
    late (converged) layers is what makes BNI discriminative.  Sampling only
    the decision zone collapses Dijkstra scores toward 1.0 for all prompts
    (adjacent late layers are always smooth) and saturates BNI ~0.9 uniformly,
    destroying the metric's ability to separate stable from unstable belief
    trajectories.
    """
    if num_layers <= 0:
        return []
    start = min(1, num_layers - 1)   # skip raw embedding output (layer 0)
    end   = num_layers - 1
    if end - start + 1 <= n_samples:
        return list(range(start, end + 1))
    step = (end - start) / (n_samples - 1)
    return sorted(set(start + round(i * step) for i in range(n_samples)))


# ── Belief extraction ──────────────────────────────────────────────────────────

@torch.no_grad()
def extract_belief_trajectory(
    model,
    tokenizer,
    prompt: str,
    layer_indices: List[int],
    top_k: int = 50,
) -> Tuple[List[torch.Tensor], List[int], List[List[int]]]:
    """
    Two-step KV-cache extraction: prefill then single-token probe pass.

    Step 1 — Prefill (no hooks):
      Run the full prompt minus the final token to build the KV cache.
      No hooks registered here — this avoids N hook callbacks on prompt tokens.

    Step 2 — Probe (hooks active):
      Run ONLY the last prompt token (the assistant-start marker, e.g.
      "<|im_start|>" or "\\n") as a single [1, 1] forward pass with the KV
      cache as context.  Hooks fire exactly once per layer, on the single
      token that directly decides the first generated token's distribution.

    Why this is the correct probe position:
      The last prompt token is the "gating" position — the model reads the
      entire user question through the KV cache and then, at this one step,
      produces the logit distribution that will be sampled as the first answer
      token.  Probing here gives us the belief trajectory ACROSS layers for
      that prediction, which is exactly what BFS/DFS/Dijkstra measure.

    Why this eliminates garbage argmax at early layers:
      In a single full-sequence pass, early-layer hooks capture the hidden
      state of the prompt's last token BEFORE it has attended to the full
      context (since the KV cache hasn't accumulated that context yet in
      intermediate representations).  With the split pass, the context is
      already materialized in the KV cache, so even layer 1 sees a richer
      representation — though early layers still tend toward noisy argmax
      because they genuinely haven't "processed" the answer yet.

    Mathematically equivalent to a full single pass (causal attention means
    the last token's output is the same either way), but:
      - Hooks only fire once per layer (not once per prompt token)
      - No sequence-position ambiguity: hs is [1, 1, D] in the probe pass
      - Explicit: the probed token IS the "decide the answer" position

    Returns
    -------
    beliefs     : list of float32 tensors [top_k] (CPU), one per sampled layer
    argmax_ids  : argmax token ID at each sampled layer
    topk_id_sets: list of top-k token ID lists, one per sampled layer
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = prompt

    inputs   = tokenizer(text, return_tensors="pt").to(model.device)
    full_ids = inputs["input_ids"]  # [1, N]

    # ── Step 1: Prefill (no hooks) ────────────────────────────────────────────
    # Build KV cache for the first N-1 tokens; the last token (assistant-start
    # marker) is processed separately in the probe step so hooks fire there only.
    past_kv = None
    if full_ids.shape[1] > 1:
        prefill_out = model(
            input_ids=full_ids[:, :-1],
            use_cache=True,
        )
        past_kv = prefill_out.past_key_values
        del prefill_out   # release intermediate activations; KV cache stays

    last_token_ids = full_ids[:, -1:]   # [1, 1] — the "decide first token" step

    # ── Step 2: Probe (single-token pass, hooks active) ───────────────────────
    lm_head    = model.lm_head if hasattr(model, "lm_head") else None
    final_norm = getattr(getattr(model, "model", None), "norm", None)
    final_layer_idx = layer_indices[-1]

    slot: Dict[int, Tuple[torch.Tensor, int, List[int]]] = {}

    def make_hook(layer_idx: int):
        def _hook(module, inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            # Single-token pass: hs is [1, 1, D] — no position ambiguity
            last = hs[:, -1:, :]
            # Apply the model's final layer norm only for the last sampled layer
            # so the probe argmax at that layer matches what generate() produces.
            if layer_idx == final_layer_idx and final_norm is not None:
                try:
                    last = final_norm(last)
                except Exception:
                    pass
            if lm_head is not None:
                logits = lm_head(last)[0, 0].float()
            else:
                logits = last[0, 0].float()

            probs             = torch.softmax(logits, dim=-1)
            topk_probs, topk_ids = torch.topk(probs, top_k)

            slot[layer_idx] = (
                topk_probs.cpu().detach(),
                topk_ids[0].item(),
                topk_ids.cpu().tolist(),
            )
            del logits, probs, topk_probs, topk_ids, last
        return _hook

    hooks = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for idx in layer_indices:
            if idx < len(model.model.layers):
                h = model.model.layers[idx].register_forward_hook(make_hook(idx))
                hooks.append(h)

    try:
        model(
            input_ids=last_token_ids,
            past_key_values=past_kv,
            use_cache=False,
        )
    finally:
        for h in hooks:
            h.remove()

    del past_kv
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Assemble results in layer_indices order ───────────────────────────────
    beliefs:      List[torch.Tensor] = []
    argmax_ids:   List[int]          = []
    topk_id_sets: List[List[int]]    = []

    for idx in layer_indices:
        if idx in slot:
            probs_cpu, top_id, top_ids = slot[idx]
            beliefs.append(probs_cpu)
            argmax_ids.append(top_id)
            topk_id_sets.append(top_ids)

    return beliefs, argmax_ids, topk_id_sets


# ── Algorithm 1: Belief BFS — Breadth-First Belief Traversal ──────────────────

def belief_bfs(
    topk_id_sets: List[List[int]],
    final_answer_id: int,
) -> float:
    """
    BFS explores the belief frontier layer-by-layer (each layer = one BFS depth).
    Score = how early the final-answer token appears in the expanding frontier.

    High score → answer token was in the model's top-k beliefs from early layers.
    Low score  → answer only emerged at the last layer (or not at all).

    Returns BFS score ∈ [0, 1].
    """
    if not topk_id_sets:
        return 0.0

    n_layers = len(topk_id_sets)
    frontier: Set[int] = set()

    for depth, ids in enumerate(topk_id_sets):
        frontier.update(ids)
        if final_answer_id in frontier:
            # depth=0 → score=1.0 (emerged at first sampled layer)
            # depth=n_layers-1 → score=1/(n_layers) (only at last layer)
            return 1.0 - (depth / max(n_layers - 1, 1))

    return 0.0  # token never appeared in top-k of any layer


# ── Algorithm 2: Belief DFS — Commitment Stability ────────────────────────────

def belief_dfs(argmax_ids: List[int]) -> float:
    """
    DFS follows the argmax token through layers and counts backtracks
    (any transition where the top token changes = a direction change).

    High score → argmax stayed committed to the same token across layers.
    Low score  → argmax zigzagged through different tokens.

    Returns DFS commitment score ∈ [0, 1].
    """
    if len(argmax_ids) < 2:
        return 1.0

    backtracks    = sum(1 for i in range(1, len(argmax_ids)) if argmax_ids[i] != argmax_ids[i - 1])
    max_backtracks = len(argmax_ids) - 1

    return 1.0 - (backtracks / max_backtracks)


# ── Algorithm 3: Dijkstra Belief Path — Confidence-Weighted Shortest Path ──────

def _jensen_shannon_div(p: torch.Tensor, q: torch.Tensor) -> float:
    """
    Approximate Jensen-Shannon divergence between two top-k probability vectors.
    Both tensors are truncated/padded to the same length and renormalized.
    """
    k = min(len(p), len(q))
    p_ = p[:k].float()
    q_ = q[:k].float()
    p_ = p_ / (p_.sum() + 1e-10)
    q_ = q_ / (q_.sum() + 1e-10)
    m  = 0.5 * (p_ + q_)
    kl_pm = (p_ * (p_ / (m + 1e-10)).log()).sum().item()
    kl_qm = (q_ * (q_ / (m + 1e-10)).log()).sum().item()
    return float(max(0.0, 0.5 * kl_pm + 0.5 * kl_qm))


def belief_dijkstra(beliefs: List[torch.Tensor]) -> float:
    """
    Dijkstra on a directed layer graph with step-1 and step-2 skip edges.

    Nodes  : layer indices 0 … K
    Edges  : (i → i+1, weight=JSD(i,i+1)) and (i → i+2, weight=JSD(i,i+2))
    Goal   : find the minimum-total-divergence path from layer 0 to layer K.

    Skip edges make the graph genuinely non-trivial — Dijkstra may prefer to
    bypass a turbulent intermediate layer if skipping it is "cheaper" (lower JSD).

    High score → smooth belief trajectory, low total divergence.
    Low score  → high total divergence (belief space was turbulent).

    Returns Dijkstra score ∈ (0, 1].
    """
    n = len(beliefs)
    if n < 2:
        return 1.0

    INF  = float("inf")
    dist = [INF] * n
    dist[0] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, 0)]

    while heap:
        cost, u = heapq.heappop(heap)
        if cost > dist[u]:
            continue
        for step in (1, 2):
            v = u + step
            if v >= n:
                break
            w        = _jensen_shannon_div(beliefs[u], beliefs[v])
            new_cost = cost + w
            if new_cost < dist[v]:
                dist[v] = new_cost
                heapq.heappush(heap, (new_cost, v))

    path_cost = dist[-1] if dist[-1] < INF else 1.0
    return 1.0 / (1.0 + path_cost)


# ── Belief Navigation Index ────────────────────────────────────────────────────

def compute_bni(bfs: float, dfs: float, dijkstra: float) -> float:
    """
    Belief Navigation Index: weighted combination of the three graph traversal scores.
    BNI ∈ [0, 1].   Higher = smoother, more stable belief trajectory.
    NOTE: BNI measures trajectory SMOOTHNESS, not truth.  A confident wrong
    answer can score just as high as a confident correct answer.
    """
    return round(0.25 * bfs + 0.35 * dfs + 0.40 * dijkstra, 4)


# ── Token display helper ───────────────────────────────────────────────────────

def _decode_token(tokenizer, token_id: int) -> str:
    """
    Return a readable string for a single token ID.

    Uses convert_ids_to_tokens (raw BPE/SPM piece) rather than decode+strip.
    decode+strip silently empties whitespace-only tokens (spaces, newlines)
    which are common as the first generated token in chat-template models.
    """
    try:
        pieces = tokenizer.convert_ids_to_tokens([token_id])
        if pieces and pieces[0] is not None:
            raw = pieces[0]
            # Normalize SentencePiece (▁) and GPT-2 BPE (Ġ) space-prefix markers
            raw = raw.replace("▁", " ").replace("Ġ", " ")
            stripped = raw.strip()
            if stripped:
                return stripped
            # Whitespace-only token — show a visible placeholder
            if " " in raw:
                return "' '"
            if "\n" in raw:
                return "'\\n'"
            return repr(raw[:8])
    except Exception:
        pass
    decoded = tokenizer.decode([token_id])
    return decoded.strip() or repr(decoded[:8])


# ── Identify critical layers from a belief trajectory ─────────────────────────

def find_commitment_layer(
    argmax_ids: List[int],
    layer_indices: List[int],
    final_id: int,
) -> Optional[int]:
    """
    Return the real layer index (into model.model.layers) where argmax first
    equals the final answer token.  Returns None if it never commits.
    """
    for pos, (tok_id, layer_idx) in enumerate(zip(argmax_ids, layer_indices)):
        if tok_id == final_id:
            return layer_idx
    return None


def find_turbulence_layer(
    beliefs: List[torch.Tensor],
    layer_indices: List[int],
) -> Optional[int]:
    """
    Return the real layer index with the highest local JSD between consecutive
    sampled layers.  This is the "roughest" point in the belief trajectory.
    """
    if len(beliefs) < 2:
        return None
    worst_jsd = -1.0
    worst_idx = None
    for i in range(len(beliefs) - 1):
        jsd = _jensen_shannon_div(beliefs[i], beliefs[i + 1])
        if jsd > worst_jsd:
            worst_jsd = jsd
            worst_idx = layer_indices[i + 1]   # the layer that *caused* the spike
    return worst_idx


# ── Hook A — Belief Anchor Hook (BAH) ─────────────────────────────────────────

class BeliefAnchorHook:
    """
    Amplifies the hidden-state component that points toward the top-token's
    unembedding direction by factor `alpha`.

    Installed on the layer where the model first commits to the answer token.
    Makes that commitment stickier — downstream layers see a stronger signal.

    alpha = 1.0  → no change (identity)
    alpha = 1.1–1.3  → gentle anchoring (recommended)
    alpha > 1.5  → aggressive (may over-sharpen, use with caution)
    """

    def __init__(self, lm_head_weight: torch.Tensor, alpha: float = 1.15):
        # lm_head_weight: [vocab_size, hidden_dim] (CPU float32)
        self.W     = lm_head_weight.float()    # keep on CPU
        self.alpha = alpha
        self._handle = None

    def _hook_fn(self, module, input, output):
        # output is a tuple; first element is the hidden-state tensor [B, T, D]
        if isinstance(output, tuple):
            hs    = output[0]
            rest  = output[1:]
        else:
            hs   = output
            rest = None

        last    = hs[:, -1:, :].float().cpu()       # [1, 1, D]
        logits  = last @ self.W.t()                  # [1, 1, vocab]
        top_id  = logits[0, 0].argmax().item()
        direction = self.W[top_id].unsqueeze(0).unsqueeze(0)  # [1, 1, D]
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Decompose last into component along direction + orthogonal remainder
        proj       = (last * direction).sum(dim=-1, keepdim=True)  # scalar projection
        last_new   = last + (self.alpha - 1.0) * proj * direction   # amplify along direction
        last_new   = last_new.to(hs.dtype).to(hs.device)

        hs_new = torch.cat([hs[:, :-1, :], last_new], dim=1)
        if rest is not None:
            return (hs_new,) + rest
        return hs_new

    def install(self, layer) -> None:
        self._handle = layer.register_forward_hook(self._hook_fn)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ── Hook B — Belief Turbulence Damper (BTD) ───────────────────────────────────

class BeliefTurbulenceDamper:
    """
    Blends a turbulent layer's output hidden state with the output of the
    *previous* layer using a mixing weight `beta`.

    h_out = (1 - beta) * h_current + beta * h_prev

    Installed on the layer with the highest local JSD in the probe pass.
    Smooths the belief spike without silencing the layer's computation.

    beta = 0.0   → no change (identity)
    beta = 0.15–0.25  → gentle smoothing (recommended)
    beta > 0.4   → heavy smoothing (may blur useful computations)
    """

    def __init__(self, beta: float = 0.20):
        self.beta     = beta
        self._prev_hs: Optional[torch.Tensor] = None
        self._handle  = None
        self._prev_handle = None

    def _prev_hook_fn(self, module, input, output):
        """Cache the previous layer's output."""
        hs = output[0] if isinstance(output, tuple) else output
        self._prev_hs = hs.detach().clone()
        return output

    def _hook_fn(self, module, input, output):
        if self._prev_hs is None:
            return output

        if isinstance(output, tuple):
            hs   = output[0]
            rest = output[1:]
        else:
            hs   = output
            rest = None

        prev = self._prev_hs.to(hs.device).to(hs.dtype)
        if prev.shape != hs.shape:
            return output   # shape mismatch guard — skip silently

        hs_smoothed = (1.0 - self.beta) * hs + self.beta * prev

        if rest is not None:
            return (hs_smoothed,) + rest
        return hs_smoothed

    def install(self, prev_layer, target_layer) -> None:
        self._prev_handle = prev_layer.register_forward_hook(self._prev_hook_fn)
        self._handle      = target_layer.register_forward_hook(self._hook_fn)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if self._prev_handle is not None:
            self._prev_handle.remove()
            self._prev_handle = None


# ── Generation (with or without hooks) ────────────────────────────────────────

@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 80,
) -> str:
    """Fast greedy generation — returns decoded new tokens only."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = prompt

    inputs     = tokenizer(text, return_tensors="pt").to(model.device)
    prefill_len = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0, prefill_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_id: str, cache_dir: str, skip_quant: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device    : {device.upper()}")

    if device == "cuda" and not skip_quant:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=True,
        )
    else:
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=True,
        )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = (
        len(model.model.layers)
        if hasattr(model, "model") and hasattr(model.model, "layers")
        else 0
    )
    print(f"  Layers    : {n_layers}")
    return model, tokenizer, n_layers


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Belief Navigation Index (BNI) + Layer Interventions — HydrusOpt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID to evaluate",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=r"D:\HydrusOPT\models",
        help="Local cache directory for model weights",
    )
    parser.add_argument(
        "--skip-quant", action="store_true",
        help="Disable 4-bit NF4 quantization (WARNING: higher VRAM usage)",
    )
    parser.add_argument(
        "--n-layers", type=int, default=8,
        help="Number of layers to sample for the belief trajectory",
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Top-k tokens to track per sampled layer",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.15,
        help="Belief Anchor Hook amplification factor (1.0=off, 1.1–1.3 recommended)",
    )
    parser.add_argument(
        "--beta", type=float, default=0.20,
        help="Belief Turbulence Damper blend weight (0.0=off, 0.15–0.25 recommended)",
    )
    parser.add_argument(
        "--no-intervene", action="store_true",
        help="Skip Phase 2 interventions — probe only",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=80,
        help="Max tokens to generate in the comparison pass",
    )
    parser.add_argument(
        "--prompts", nargs="+", default=None,
        help="Custom prompts to evaluate (overrides defaults)",
    )
    args = parser.parse_args()

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS

    print(f"\n{'='*68}")
    print(f"  Belief Navigation Index (BNI) + Layer Interventions")
    print(f"  HydrusOpt — graph-traversal hallucination probe & repair")
    print(f"{'='*68}")
    print(f"  Model         : {args.model}")

    model, tokenizer, n_layers = load_model(args.model, args.cache_dir, args.skip_quant)

    # lm_head weight reference for BeliefAnchorHook (CPU copy, used only by BAH)
    lm_head_weight_cpu: Optional[torch.Tensor] = None
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        lm_head_weight_cpu = model.lm_head.weight.detach().float().cpu()

    layer_indices = _sample_layer_indices(n_layers, n_samples=args.n_layers)
    print(f"  Sampled layers: {layer_indices}")
    print(f"  Top-k         : {args.top_k}  |  alpha (BAH): {args.alpha}  |  beta (BTD): {args.beta}")
    print(f"  Prompts       : {len(prompts)}\n")

    results: List[Dict[str, Any]] = []

    for i, prompt in enumerate(prompts, 1):
        short_prompt = (prompt[:65] + "…") if len(prompt) > 65 else prompt
        print(f"  [{i:02}/{len(prompts):02}] {short_prompt}")

        # ── PHASE 1: Probe ─────────────────────────────────────────────────────
        beliefs, argmax_ids, topk_id_sets = extract_belief_trajectory(
            model, tokenizer, prompt, layer_indices, top_k=args.top_k
        )

        final_id    = argmax_ids[-1]
        final_token = _decode_token(tokenizer, final_id)

        # Shannon entropy of the final-layer top-k distribution.
        # Measures output CONFIDENCE (not trajectory smoothness like BNI).
        # Low entropy → confident (possibly confidently wrong).
        # High entropy → spread beliefs (model is genuinely unsure).
        final_probs   = beliefs[-1].float()
        final_probs   = final_probs / (final_probs.sum() + 1e-10)
        final_entropy = float(-(final_probs * final_probs.clamp_min(1e-10).log()).sum().item())

        bfs_score  = belief_bfs(topk_id_sets, final_id)
        dfs_score  = belief_dfs(argmax_ids)
        dijk_score = belief_dijkstra(beliefs)
        bni_before = compute_bni(bfs_score, dfs_score, dijk_score)
        label_b    = "STABLE" if bni_before >= 0.55 else ("UNCERTAIN" if bni_before >= 0.35 else "UNSTABLE")

        commit_layer     = find_commitment_layer(argmax_ids, layer_indices, final_id)
        turbulence_layer = find_turbulence_layer(beliefs, layer_indices)

        path_tokens = [_decode_token(tokenizer, t) for t in argmax_ids]
        path_str = " → ".join(repr(t) for t in path_tokens[:4]) + (" → …" if len(path_tokens) > 4 else "")

        print(f"         Token (argmax) : {repr(final_token)}  H(p)={final_entropy:.3f}")
        print(f"         Argmax path    : {path_str}")
        print(f"         Commit layer   : {commit_layer}  |  Turbulence layer: {turbulence_layer}")
        print(f"         BFS  : {bfs_score:.4f}  DFS  : {dfs_score:.4f}  Dijkstra : {dijk_score:.4f}")
        print(f"         BNI (before)   : {bni_before:.4f}  [{label_b}]")

        del beliefs, argmax_ids, topk_id_sets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── PHASE 2: Intervene ─────────────────────────────────────────────────
        bah = None
        btd = None
        baseline_answer   = ""
        intervened_answer = ""
        bni_after         = None
        label_a           = ""
        hooks_installed   = []

        if not args.no_intervene and hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers

            final_layer = layer_indices[-1]  # last sampled layer index

            # Hook A — Belief Anchor Hook on the commitment layer
            # Skip if commit is the final sampled layer — no downstream layers
            # to reinforce; amplifying the last layer's residual can corrupt
            # the token distribution (e.g. '7' → '79').
            if (
                commit_layer is not None
                and commit_layer != final_layer
                and lm_head_weight_cpu is not None
                and args.alpha != 1.0
            ):
                bah = BeliefAnchorHook(lm_head_weight_cpu, alpha=args.alpha)
                bah.install(layers[commit_layer])
                hooks_installed.append(f"BAH@L{commit_layer}")

            # Hook B — Belief Turbulence Damper on the turbulence layer
            # Skip if turbulence is the final sampled layer — blending L(n-1)
            # into the final output layer destabilizes the decision directly.
            if (
                turbulence_layer is not None
                and turbulence_layer != final_layer
                and turbulence_layer > 0
                and args.beta > 0.0
            ):
                btd = BeliefTurbulenceDamper(beta=args.beta)
                btd.install(layers[turbulence_layer - 1], layers[turbulence_layer])
                hooks_installed.append(f"BTD@L{turbulence_layer}")

            # Baseline generation (no hooks installed yet for baseline — generate first)
            # Order: we want baseline WITHOUT hooks, then install hooks for intervened.
            # But hooks are already installed above. So:
            #   - remove hooks, generate baseline, reinstall hooks, generate intervened
            if bah:
                bah.remove()
            if btd:
                btd.remove()

            baseline_answer = generate_answer(model, tokenizer, prompt, args.max_new_tokens)

            # Reinstall and generate with interventions
            if bah:
                bah.install(layers[commit_layer])
            if btd:
                btd.install(layers[turbulence_layer - 1], layers[turbulence_layer])

            intervened_answer = generate_answer(model, tokenizer, prompt, args.max_new_tokens)

            # Remove all hooks immediately after use
            if bah:
                bah.remove()
            if btd:
                btd.remove()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Re-probe BNI with the intervened generation (another single forward pass)
            if hooks_installed:
                if bah:
                    bah.install(layers[commit_layer])
                if btd:
                    btd.install(layers[turbulence_layer - 1], layers[turbulence_layer])

                beliefs2, argmax2, topk2 = extract_belief_trajectory(
                    model, tokenizer, prompt, layer_indices, top_k=args.top_k
                )
                bfs2  = belief_bfs(topk2, argmax2[-1])
                dfs2  = belief_dfs(argmax2)
                dijk2 = belief_dijkstra(beliefs2)
                bni_after = compute_bni(bfs2, dfs2, dijk2)
                label_a   = "STABLE" if bni_after >= 0.55 else ("UNCERTAIN" if bni_after >= 0.35 else "UNSTABLE")

                if bah:
                    bah.remove()
                if btd:
                    btd.remove()

                del beliefs2, argmax2, topk2
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if hooks_installed:
                delta_str = ""
                if bni_after is not None:
                    delta = bni_after - bni_before
                    sign  = "+" if delta >= 0 else ""
                    delta_str = f"  ΔBNI: {sign}{delta:.4f}"
                print(f"         Hooks          : {', '.join(hooks_installed)}{delta_str}")
                if bni_after is not None:
                    print(f"         BNI (after)    : {bni_after:.4f}  [{label_a}]")
                base_short = (baseline_answer[:70] + "…") if len(baseline_answer) > 70 else baseline_answer
                int_short  = (intervened_answer[:70] + "…") if len(intervened_answer) > 70 else intervened_answer
                print(f"         Baseline answer: {repr(base_short)}")
                print(f"         Intervened     : {repr(int_short)}")
            else:
                print(f"         [no hooks applicable — BAH or BTD conditions not met]")

        print()

        results.append({
            "prompt":     short_prompt,
            "token":      final_token,
            "bfs":        bfs_score,
            "dfs":        dfs_score,
            "dijkstra":   dijk_score,
            "bni_before": bni_before,
            "bni_after":  bni_after,
            "label":      label_b,
            "entropy":    final_entropy,
            "hooks":      ", ".join(hooks_installed) if hooks_installed else "—",
        })

    # ── Summary table ──────────────────────────────────────────────────────────
    W = 40
    print(f"\n{'='*76}")
    print(f"  BNI SUMMARY — BEFORE / AFTER INTERVENTIONS")
    print(f"{'='*76}")
    print(f"  {'Prompt':<{W}} {'BFS':>5} {'DFS':>5} {'Dijk':>6} {'BNI-B':>6} {'BNI-A':>6} {'ΔBNI':>6} {'H(p)':>5}  Hooks")
    print(f"  {'-'*W} {'-'*5} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5}  -----")
    for r in results:
        after_str = f"{r['bni_after']:.4f}" if r["bni_after"] is not None else "  n/a "
        delta     = (r["bni_after"] - r["bni_before"]) if r["bni_after"] is not None else None
        delta_str = (f"{delta:+.4f}" if delta is not None else "     ")
        ent_str   = f"{r.get('entropy', 0.0):.2f}"
        print(
            f"  {r['prompt']:<{W}} "
            f"{r['bfs']:>5.3f} {r['dfs']:>5.3f} {r['dijkstra']:>6.4f} "
            f"{r['bni_before']:>6.4f} {after_str:>6} {delta_str:>6} {ent_str:>5}  {r['hooks']}"
        )

    avg_before = sum(r["bni_before"] for r in results) / max(len(results), 1)
    afters     = [r["bni_after"] for r in results if r["bni_after"] is not None]
    avg_after  = sum(afters) / max(len(afters), 1) if afters else None

    print(f"\n  Average BNI before: {avg_before:.4f}")
    if avg_after is not None:
        sign = "+" if avg_after - avg_before >= 0 else ""
        print(f"  Average BNI after : {avg_after:.4f}  ({sign}{avg_after - avg_before:.4f})")
    print(f"\n  Legend:")
    print(f"    BNI ≥ 0.55 → STABLE    — smooth belief traversal, stable argmax")
    print(f"    BNI ≥ 0.35 → UNCERTAIN — moderate trajectory turbulence")
    print(f"    BNI < 0.35 → UNSTABLE  — high turbulence, frequent argmax flipping")
    print(f"    IMPORTANT: BNI measures trajectory SMOOTHNESS, not factual correctness.")
    print(f"               A confident wrong answer scores just as high as a correct one.")
    print(f"               Use H(p) as a complementary confidence signal.")
    print(f"    H(p) — Shannon entropy of the top-{args.top_k} output distribution at final layer.")
    print(f"           Low  H(p) = model is confident (does not imply correct answer).")
    print(f"           High H(p) = model is uncertain / beliefs spread across many tokens.")
    print(f"\n  Hooks:")
    print(f"    BAH@Ln — Belief Anchor Hook at layer n (amplifies commitment, alpha={args.alpha})")
    print(f"    BTD@Ln — Belief Turbulence Damper at layer n (blends with prev layer, beta={args.beta})\n")


if __name__ == "__main__":
    main()
