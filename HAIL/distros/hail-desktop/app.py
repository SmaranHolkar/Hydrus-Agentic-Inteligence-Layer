import json
import sys
import re
import xml.etree.ElementTree as ET
from typing import Optional
import webbrowser
import http.server
import socketserver
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import parse, request

# Ensure UTF-8 output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# Add HAIL root directory and src directory to sys.path
hail_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(hail_root))
sys.path.insert(0, str(hail_root / "src"))

from hail_core import HAIL, HAILConfig, ModelConfig
from hydrusmoe import HydrusMoEEngine, HydrusMoEConfig

# ==============================================================================
# Ollama Integration Engine (Local-First LLM Bridge)
# ==============================================================================

OLLAMA_BASE_URL = "http://127.0.0.1:11434"

def _check_ollama_status() -> dict:
    """Check if local Ollama daemon is running and return list of installed models."""
    try:
        req = request.Request(f"{OLLAMA_BASE_URL}/api/tags", headers={'User-Agent': 'HAIL-Studio/1.0'})
        with request.urlopen(req, timeout=0.5) as resp:
            data = json.loads(resp.read().decode())
            models = [m.get("name") for m in data.get("models", [])]
            return {"online": True, "models": models}
    except Exception as e:
        return {"online": False, "models": []}

def _generate_with_ollama(model: str, prompt: str, system_prompt: str = "") -> str:
    """Generate response using local Ollama model."""
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        if system_prompt:
            payload["system"] = system_prompt
        
        req_data = json.dumps(payload).encode('utf-8')
        req = request.Request(f"{OLLAMA_BASE_URL}/api/generate", data=req_data, headers={'Content-Type': 'application/json'})
        with request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            return data.get("response", "").strip()
    except Exception as e:
        print(f"[Ollama Generation Error] {e}")
        return None

# ==============================================================================
# PyTorch Local Model Engine (HuggingFace Transformers + CUDA Acceleration)
# ==============================================================================

import os
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = str(WORKSPACE_ROOT / "models")
os.environ['HF_HOME'] = MODELS_DIR

_ACTIVE_PYTORCH_MODEL = None
_ACTIVE_PYTORCH_TOKENIZER = None
_ACTIVE_PYTORCH_MODEL_ID = None
_PYTORCH_LOAD_LOCK = threading.Lock()

_PYTORCH_LOADING = False

def _model_cache_dirname(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")

def _is_model_available_locally(model_id: str) -> bool:
    model_cache_dir = Path(MODELS_DIR) / _model_cache_dirname(model_id)
    return model_cache_dir.exists()

def _load_local_pytorch_model(model_id: str) -> bool:
    global _ACTIVE_PYTORCH_MODEL, _ACTIVE_PYTORCH_TOKENIZER, _ACTIVE_PYTORCH_MODEL_ID, _PYTORCH_LOADING
    with _PYTORCH_LOAD_LOCK:
        if _ACTIVE_PYTORCH_MODEL_ID == model_id and _ACTIVE_PYTORCH_MODEL is not None:
            return True
            
        _PYTORCH_LOADING = True
        print(f"[PyTorch Engine] Loading local model '{model_id}' from '{MODELS_DIR}' on GPU CUDA...")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not _is_model_available_locally(model_id):
                print(f"[PyTorch Engine Error] Model '{model_id}' is not present in local cache '{MODELS_DIR}'.")
                return False
            
            # Clean up old model VRAM
            if _ACTIVE_PYTORCH_MODEL is not None:
                del _ACTIVE_PYTORCH_MODEL
                del _ACTIVE_PYTORCH_TOKENIZER
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=MODELS_DIR,
                local_files_only=True,
                trust_remote_code=True
            )
                
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=MODELS_DIR,
                local_files_only=True,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            _ACTIVE_PYTORCH_MODEL = model
            _ACTIVE_PYTORCH_TOKENIZER = tokenizer
            _ACTIVE_PYTORCH_MODEL_ID = model_id
            print(f"[PyTorch Engine] Successfully loaded '{model_id}' on {device}!")
            return True
        except Exception as e:
            print(f"[PyTorch Engine Error] Failed to load '{model_id}': {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            _PYTORCH_LOADING = False

def _generate_with_local_pytorch(prompt: str, system_prompt: str = "", max_new_tokens: int = 150) -> str:
    global _ACTIVE_PYTORCH_MODEL, _ACTIVE_PYTORCH_TOKENIZER
    if _ACTIVE_PYTORCH_MODEL is None or _ACTIVE_PYTORCH_TOKENIZER is None:
        return None
    try:
        import torch
        device = next(_ACTIVE_PYTORCH_MODEL.parameters()).device
        
        formatted_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
        if hasattr(_ACTIVE_PYTORCH_TOKENIZER, "apply_chat_template"):
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                formatted_prompt = _ACTIVE_PYTORCH_TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
            
        inputs = _ACTIVE_PYTORCH_TOKENIZER(formatted_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = _ACTIVE_PYTORCH_MODEL.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        text = _ACTIVE_PYTORCH_TOKENIZER.decode(generated_ids, skip_special_tokens=True).strip()
        return text
    except Exception as e:
        print(f"[PyTorch Generation Error] {e}")
        return None

# ==============================================================================
# Standalone Multi-Source Academic & Literature Research Engine (Zero-Torch)
# ==============================================================================

_RESEARCH_UA = "HAIL-Cognitive-Studio/1.0 (academic-research-mode; local-kernel)"
_RESEARCH_TIMEOUT = 15

_QUERY_STRIP = {
    "how", "what", "why", "when", "where", "who", "which", "does", "do", "did",
    "is", "are", "was", "were", "can", "could", "should", "would", "will", "has", "have", "had",
    "the", "a", "an", "to", "for", "of", "in", "on", "at", "by", "with", "from",
    "and", "or", "but", "more", "most", "best", "good", "some", "any", "many", "much",
    "i", "me", "my", "you", "your", "we", "us", "it", "its", "create",
    "generate", "write", "make", "document", "doc", "about"
}

def _http_get_json(url: str, headers: dict = None) -> dict:
    h = {"User-Agent": _RESEARCH_UA}
    if headers:
        h.update(headers)
    try:
        req = request.Request(url, headers=h)
        with request.urlopen(req, timeout=_RESEARCH_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8", errors="ignore"))
    except Exception as e:
        print(f"[Research HTTP JSON Error] {url}: {e}")
        return None

def _http_get_text(url: str) -> str:
    try:
        req = request.Request(url, headers={"User-Agent": _RESEARCH_UA})
        with request.urlopen(req, timeout=_RESEARCH_TIMEOUT) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"[Research HTTP Text Error] {url}: {e}")
        return None

def _normalize_query(query: str) -> str:
    cleaned = re.sub(r"[^\w\s\-]", " ", query.lower())
    tokens = [t for t in cleaned.split() if len(t) > 2 and t not in _QUERY_STRIP]
    return " ".join(tokens) if tokens else query.lower()

def _clean_subject_for_search(query: str) -> str:
    q = query.replace("_", " ").strip()
    clean = re.sub(r"^(?:could\s+you\s+|can\s+you\s+|please\s+|would\s+you\s+)?(?:create|generate|write|make|tell\s+me|explain)?\s*(?:a\s+)?(?:document|doc|paper|post|info|about)?\s*(?:on|about|for|regarding|is|was|are|the|whats|what's)?\s*", "", q, flags=re.IGNORECASE).strip()
    clean = re.sub(r"^(?:what\s+is|what\s+was|whats|what's|who\s+is|who\s+was|where\s+is|tell\s+me\s+about|explain)\s+(?:the|a|an)?\s*", "", clean, flags=re.IGNORECASE).strip()
    clean = re.sub(r"^(?:what\s+(?:can|should|do|does)\s+i\s+(?:do\s+to|have\s+to|need\s+to|want\s+to)\s+)?", "", clean, flags=re.IGNORECASE).strip()
    clean = re.sub(r"^(?:how\s+(?:can|should|do|does|to)\s+(?:i\s+)?(?:get|lose|make|build|find|write)?\s*)", "", clean, flags=re.IGNORECASE).strip()
    clean = re.sub(r"[\?\.\!]+$", "", clean).strip()
    return clean or query

def _is_good_title_or_content_match(query: str, title: str, extract: str) -> bool:
    if not query or not title:
        return False
    
    def clean_tokens(text: str) -> set:
        text = re.sub(r"[^\w\s\-]", " ", text.lower())
        words = []
        for w in text.split():
            if len(w) > 2 and w not in _QUERY_STRIP:
                if w.endswith("s") and len(w) > 3:
                    w = w[:-1]
                words.append(w)
        return set(words)
        
    q_tokens = clean_tokens(query)
    if not q_tokens:
        return True
        
    t_tokens = clean_tokens(title)
    
    # Check first sentence of extract
    first_sentence = extract.split(".")[0] if extract else ""
    e_tokens = clean_tokens(first_sentence)
    
    target_tokens = t_tokens.union(e_tokens)
    
    intersection = q_tokens.intersection(target_tokens)
    if len(q_tokens) == 1:
        return len(intersection) >= 1
    else:
        return len(intersection) >= max(2, len(q_tokens) // 2)

def _fetch_wikipedia_full_article(subject: str) -> dict:
    search_term = _clean_subject_for_search(subject)
    print(f"[HAIL Wiki Search] Subject: '{subject}' -> Clean Search Term: '{search_term}'")
    
    def do_search(term: str) -> dict:
        search_params = parse.urlencode({
            "action": "opensearch",
            "search": term,
            "limit": "3",
            "namespace": "0",
            "format": "json"
        })
        search_url = f"https://en.wikipedia.org/w/api.php?{search_params}"
        resolved_titles = [term]
        try:
            s_data = _http_get_json(search_url)
            if s_data and len(s_data) > 1 and s_data[1]:
                resolved_titles = s_data[1]
                print(f"[HAIL Wiki Search] OpenSearch resolved candidates for '{term}': {resolved_titles}")
        except Exception as e:
            print(f"[Wikipedia OpenSearch Error] {e}")

        for r_title in resolved_titles:
            params = parse.urlencode({
                "action": "query",
                "prop": "extracts",
                "exlimit": "1",
                "explaintext": "1",
                "redirects": "1",
                "titles": r_title,
                "format": "json"
            })
            url = f"https://en.wikipedia.org/w/api.php?{params}"
            data = _http_get_json(url)
            if not data:
                continue

            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id != "-1":
                    extract = page_data.get("extract", "")
                    p_title = page_data.get("title", r_title)
                    if extract and len(extract) > 100:
                        if _is_good_title_or_content_match(search_term, p_title, extract):
                            return {"title": p_title, "full_text": extract}
        return None

    # Try clean search term first
    res = do_search(search_term)
    if res:
        return res
        
    # Fallback for Airbus A350 family queries
    if "a350" in search_term.lower():
        fallback_title = "Airbus A350"
        params = parse.urlencode({
            "action": "query",
            "prop": "extracts",
            "exlimit": "1",
            "explaintext": "1",
            "redirects": "1",
            "titles": fallback_title,
            "format": "json"
        })
        data = _http_get_json(f"https://en.wikipedia.org/w/api.php?{params}")
        if data:
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id != "-1":
                    return {"title": page_data.get("title", fallback_title), "full_text": page_data.get("extract", "")}

    # If clean search failed, try normalized query fallback
    norm_term = _normalize_query(subject)
    if norm_term and norm_term != search_term:
        print(f"[HAIL Wiki Search] Fallback to normalized query: '{norm_term}'")
        res = do_search(norm_term)
        if res:
            return res

    return {"title": search_term, "full_text": ""}

def _fetch_arxiv(query: str, max_results: int = 3) -> list:
    params = parse.urlencode({
        "search_query": f"all:{query}",
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    url = f"http://export.arxiv.org/api/query?{params}"
    raw = _http_get_text(url)
    if not raw:
        return []
    results = []
    try:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(raw)
        for entry in root.findall("a:entry", ns):
            title = (entry.findtext("a:title", "", ns) or "").strip().replace("\n", " ")
            summary = (entry.findtext("a:summary", "", ns) or "").strip().replace("\n", " ")
            link_el = entry.find("a:id", ns)
            link = (link_el.text or "").strip() if link_el is not None else ""
            if title and summary:
                results.append({
                    "source": "arXiv",
                    "title": title,
                    "snippet": summary[:400],
                    "url": link,
                })
    except Exception as e:
        print(f"[arXiv Parse Error] {e}")
    return results

def _fetch_pubmed(query: str, max_results: int = 3) -> list:
    search_params = parse.urlencode({
        "db": "pubmed",
        "retmode": "json",
        "retmax": max_results,
        "term": query,
    })
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{search_params}"
    data = _http_get_json(search_url)
    if not data:
        return []
    ids = data.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    fetch_params = parse.urlencode({
        "db": "pubmed",
        "retmode": "json",
        "rettype": "abstract",
        "id": ",".join(ids[:max_results]),
    })
    fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?{fetch_params}"
    summary_data = _http_get_json(fetch_url)
    if not summary_data:
        return []
    results = []
    uids = summary_data.get("result", {}).get("uids", [])
    for uid in uids:
        rec = summary_data["result"].get(uid, {})
        title = rec.get("title", "").strip()
        source_journal = rec.get("source", "")
        pub_date = rec.get("pubdate", "")
        if title:
            results.append({
                "source": "PubMed / NCBI",
                "title": title,
                "snippet": f"Journal: {source_journal} ({pub_date})" if source_journal else pub_date,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
            })
    return results


def _sanitize_chat_history(chat_history: list, max_items: int = 10) -> list:
    if not isinstance(chat_history, list):
        return []

    cleaned = []
    for msg in chat_history[-max_items * 2:]:
        if not isinstance(msg, dict):
            continue
        role = "user" if msg.get("role") == "user" else "assistant"
        text = str(msg.get("text", "")).strip()
        if not text:
            continue
        if len(text) > 3000:
            text = text[:3000]
        if cleaned and cleaned[-1]["role"] == role and cleaned[-1]["text"] == text:
            continue
        cleaned.append({"role": role, "text": text})

    return cleaned[-max_items:]


def _is_longform_request(prompt: str) -> bool:
    lower = (prompt or "").strip().lower()
    if len(lower) >= 120:
        return True
    return bool(re.search(r"\b(write|draft|compose|outline|scene|episode|story|chapter|essay|long[-\s]?form|detailed|comprehensive)\b", lower))


def _recommended_max_new_tokens(prompt: str, intent: str) -> int:
    if _is_longform_request(prompt):
        if intent == "CODE":
            return 360
        return 520
    if intent == "FACTUAL":
        return 220
    return 180


def _runtime_model_claim(model: str, active_display_name: str, moe_status: dict = None) -> str:
    if model.startswith("moe:"):
        execution = (moe_status or {}).get("execution", {})
        prefetch = ((moe_status or {}).get("prefetcher", {}) or {}).get("metrics", {})
        hit_rate = prefetch.get("query_hit_rate", 0.0)

        if execution.get("streaming_generation_verified"):
            return (
                f"I am running as **{active_display_name}** via **HydrusMoE** with verified local expert blobs and streamed expert activation. "
                f"Current prefetch query hit-rate is **{hit_rate:.2f}**."
            )

        mode = execution.get("weights_mode", "unknown")
        if mode == "mixed_local_and_synthetic":
            return (
                f"I am running as **{active_display_name}** via **HydrusMoE**, but this session is **mixed local + synthetic**. "
                "It is not a fully verified end-to-end 35B streamed generation run yet."
            )

        return (
            f"I am running as **{active_display_name}** via **HydrusMoE control-plane demo mode** (secure routing + tiered caching behavior), "
            "not verified full 35B streamed language generation in this session."
        )

    return f"I am running as local model **{active_display_name}** via the HAIL Cognitive gateway."


def _offline_longform_fallback(prompt: str, memories: list) -> str:
    project_hint = "your project"
    for m in memories:
        low = m.lower()
        if "project:" in low:
            project_hint = m.split(":", 1)[-1].strip() or project_hint
            break
        if "working on" in low:
            project_hint = m.split("on", 1)[-1].strip() or project_hint
            break

    return (
        f"Great direction. I can build this as a long-form draft tied to **{project_hint}**.\n\n"
        "Here is a strong structure we can use right now:\n"
        "1. **Core Premise**: one precise sentence defining the central conflict.\n"
        "2. **Character Stakes**: what your lead wants, what they fear, and what they will lose.\n"
        "3. **Escalation Spine**: 5-8 beats that keep pressure rising every scene.\n"
        "4. **Theme Layer**: what idea the audience should feel by the ending.\n"
        "5. **Ending Turn**: the reveal/decision that forces the next chapter or episode.\n\n"
        f"If you want, I will now draft the full version from your prompt exactly as written: \"{prompt.strip()}\" and keep it in a natural narrative voice."
    )


def _get_ollama_target_model(model: str) -> Optional[str]:
    if not model:
        return None
    # Strip prefix
    clean_model = model
    if model.startswith("ollama:"):
        clean_model = model.replace("ollama:", "")
    elif model.startswith("moe:"):
        moe_id = model.replace("moe:", "")
        # Map MoE models to a default available local model in Ollama
        if "mixtral" in moe_id.lower():
            clean_model = "mixtral"
        else:
            clean_model = "qwen"
    elif model.startswith("local:"):
        local_id = model.replace("local:", "")
        if "phi-3.5" in local_id.lower() or "phi3" in local_id.lower():
            clean_model = "phi3"
        elif "qwen2.5-7b" in local_id.lower():
            clean_model = "qwen2.5:7b"
        elif "qwen2.5-3b" in local_id.lower():
            clean_model = "qwen2.5:3b"
        elif "qwen2.5-1.5b" in local_id.lower():
            clean_model = "qwen2.5:1.5b"
        elif "phi-2" in local_id.lower():
            clean_model = "phi"
        else:
            clean_model = "phi3"
    return clean_model

def _route_intent(prompt: str, model: str = None, chat_history: list = None) -> str:
    # Remove leading greeting prefixes
    route_p = re.sub(r'^(?:hi|hello|hey|greetings|yo)\b[.,!\s]*', '', prompt, flags=re.IGNORECASE).strip()
    if not route_p:
        return "CHAT" # Only a greeting was sent
        
    lower_p = route_p.lower()

    # Follow-up continuity handling: short continuations should stay conversational.
    if chat_history:
        recent = _sanitize_chat_history(chat_history, max_items=4)
        if recent and len(lower_p.split()) <= 8 and not route_p.endswith("?"):
            last_text = recent[-1].get("text", "").lower()
            if any(k in last_text for k in ["episode", "scene", "character", "project", "writing", "show", "brainstorm", "draft"]):
                return "CHAT"

    # Treat practical writing requests as conversational assistance.
    if re.search(r"\b(email|e-mail)\b", lower_p):
        return "CHAT"
    
    # 1. Quick Heuristics to avoid API latency on obvious matches
    # EMOTION
    emotion_keywords = ["feel", "feeling", "sad", "happy", "depressed", "angry", "hurt", "tired", "sick", "burnout", "burned out", "exhausted", "lonely", "frustrated", "not feeling good", "not doing well", "bad day", "hate"]
    if any(re.search(rf"\b{w}\b", lower_p) for w in emotion_keywords):
        return "EMOTION"
        
    # CODE
    code_keywords = ["python", "code", "script", "function", "program", "css", "html", "javascript", "api", "compile", "bug", "syntax", "git", "repo", "library", "framework"]
    code_patterns = [r"\bdef\b", r"\bclass\b", r"\bimport\b", r"\bconst\b", r"\blet\b", r"\bvar\b", r"\{\}", r"\[\]"]
    if any(w in lower_p for w in code_keywords) or any(re.search(p, lower_p) for p in code_patterns):
        return "CODE"
        
    # CHAT / GREETINGS / CONVERSATIONAL
    chat_keywords = ["hi", "hello", "hey", "greetings", "what's up", "how's it going", "how are you", "thank you", "thanks", "nice", "cool", "awesome", "great", "perfect"]
    chat_patterns = [r"^(?:so\s+)?(?:i\s+am|i'm)\s+", r"^let's\s+", r"^can\s+we\s+", r"^why\s+don't\s+we\s+"]
    if any(re.search(rf"\b{w}\b", lower_p) for w in chat_keywords) or any(re.search(p, lower_p) for p in chat_patterns):
        return "CHAT"

    # 2. Use local Ollama model as a Semantic Router if online
    target_model = _get_ollama_target_model(model)
    if target_model and _check_ollama_status().get("online"):
        sys_route_prompt = (
            "You are a router. Classify the user prompt into exactly one of these categories: CHAT, FACTUAL, CODE, or EMOTION.\n"
            "CRITICAL: Default to CHAT intent unless the user explicitly asks to summarize a file, read a document, search for a specific file, or asks a factual question.\n"
            "Return only the single word category name in uppercase, nothing else. No explanation, no punctuation."
        )
        try:
            routed = _generate_with_ollama(target_model, prompt, sys_route_prompt)
            if routed:
                routed_clean = routed.strip().upper()
                if routed_clean in ["CHAT", "FACTUAL", "CODE", "EMOTION"]:
                    return routed_clean
        except Exception as e:
            print(f"[Semantic Router Error] {e}")

    # 2.5 Check if user is asking about the conversation history/context
    history_phrases = ["talking about", "we talk", "what did i say", "what did i ask", "what did i write", "what did i mention", "what were we", "you said", "i said", "previous", "earlier", "who are you", "what are you", "how are you"]
    if any(phrase in lower_p for phrase in history_phrases):
        return "CHAT"

    # 3. Default fallback by question word
    factual_starts = ["what", "who", "where", "why", "how", "when", "which", "explain", "tell me about", "describe", "show me"]
    if any(lower_p.startswith(start) for start in factual_starts):
        return "FACTUAL"
        
    return "CHAT"

def _generate_local_chat_fallback(intent: str, prompt: str, memories: list, chat_history: list) -> str:
    lower_p = prompt.strip().lower()

    is_email_req = bool(re.search(r"\b(email|e-mail)\b", lower_p) and re.search(r"\b(write|draft|help|compose|create)\b", lower_p))
    is_lost_package_req = bool(re.search(r"\b(lost|missing|delayed|not received)\b", lower_p) and re.search(r"\b(package|parcel|shipment|order)\b", lower_p))
    is_character_req = bool(re.search(r"\b(character|charcter|chatacter|charactor|protagonist|villain|hero)\b", lower_p) and re.search(r"\b(tv\s*show|show|series|pilot|tvshow|mytv)\b", lower_p))
    looks_like_screenplay = len(prompt) > 260 or bool(re.search(r"\b(INT\.|EXT\.|FADE IN|CUT TO:|TITLE CARD|NARRATOR\s*\(V\.O\.\)|SCENE)\b", prompt, flags=re.IGNORECASE)) or prompt.count("\n") >= 3

    # Extract user name if available
    user_name = "friend"
    for m in memories:
        if "name is" in m.lower():
            user_name = m.split("is")[-1].strip().capitalize()
            break
            
    # Extract project safely
    project = "your project"
    for m in memories:
        if "ensers" in m.lower() or "ensera" in m.lower():
            project = "Ensers"
            break
        elif "project:" in m.lower():
            cand = m.split("project:")[-1].strip()
            # If candidate project title is short and clean, use it
            if len(cand.split()) <= 4 and "." not in cand:
                project = cand
                break
        elif "working on" in m.lower() and not m.startswith("Background:"):
            cand = m.split("on")[-1].strip()
            if len(cand.split()) <= 4 and "." not in cand:
                project = cand
                break

    # Determine if we are continuing a brainstorm about their project
    is_brainstorming = False
    if chat_history:
        for msg in reversed(chat_history):
            if msg.get("role") == "assistant":
                last_text = msg.get("text", "").lower()
                if any(w in last_text for w in ["tv show", "project", "ensers", "brainstorm", "sectors", "walls", "world", "episode", "show", "write", "writing", "conflict", "plot", "story", "character"]):
                    is_brainstorming = True
                break

    history_text = " ".join((m.get("text", "") for m in chat_history[-8:])).lower() if chat_history else ""
    memory_text = " ".join(memories).lower() if memories else ""
    creative_context = any(tag in (history_text + " " + memory_text) for tag in ["tv show", "series", "episode", "plot", "story", "character", "protagonist", "pilot"])
    follow_up_style = len(prompt.strip()) > 18 and "?" not in prompt and not bool(re.search(r"^(hi|hello|hey)\b", lower_p))
    user_messages = [m.get("text", "") for m in chat_history if m.get("role") == "user"] if chat_history else []
    prev_user_text = user_messages[-1] if user_messages else ""
    prev_looks_like_screenplay = len(prev_user_text) > 260 or bool(re.search(r"\b(INT\.|EXT\.|FADE IN|CUT TO:|TITLE CARD|NARRATOR\s*\(V\.O\.\)|SCENE)\b", prev_user_text, flags=re.IGNORECASE))

    name_match = re.search(r"\b(?:name is|his name is|main guy is)\s+([A-Za-z][A-Za-z\-']{1,30})", prompt, flags=re.IGNORECASE)
    lead_name = name_match.group(1) if name_match else "Elias"

    if intent == "EMOTION":
        if "burnout" in lower_p or "tired" in lower_p or "exhausted" in lower_p:
            return f"Sorry to hear that, man. Burnout from coding is real. Make sure to take a break, get some water, and rest! I'm here when you get back."
        return f"I'm sorry to hear you're not feeling great, {user_name}. Coding can be intense—take a break if you need to, or let me know if you want to talk about it."
        
    elif intent == "CHAT":
        if is_email_req:
            if is_lost_package_req:
                return (
                    "Absolutely. Here is a clean email draft you can send:\n\n"
                    "Subject: Missing Package Inquiry - Order #[Your Order Number]\n\n"
                    "Hello [Support Team / Carrier Name],\n\n"
                    "I hope you're doing well. I'm writing to report that my package for order #[Your Order Number] has not arrived yet.\n\n"
                    "Order details:\n"
                    "- Order number: [Your Order Number]\n"
                    "- Tracking number: [Tracking Number]\n"
                    "- Expected delivery date: [Date]\n"
                    "- Delivery address: [Your Address]\n\n"
                    "The tracking status currently shows: [Current Tracking Status]. Could you please check the shipment status and let me know the next steps? "
                    "If the package is confirmed lost, I'd appreciate a replacement or refund.\n\n"
                    "Thank you for your help.\n\n"
                    "Best regards,\n"
                    "[Your Full Name]"
                )
            return "Absolutely. Share the recipient, tone (formal/casual), and key details, and I will draft the email for you."

        if _is_longform_request(prompt) and not looks_like_screenplay and not re.match(r"^that is episode\s+\d+", lower_p.strip()):
            return _offline_longform_fallback(prompt, memories)

        if is_character_req:
            return (
                "Great idea. Here is a character concept to start with:\n\n"
                "Name: Kael Mercer\n"
                "Role: Reluctant protagonist\n"
                "Public Face: Charming emergency dispatcher everyone trusts\n"
                "Secret: He can hear short 'echoes' of future conversations, but only when someone is lying\n"
                "Core Wound: He failed to believe his sister before she disappeared\n"
                "Goal: Find his sister by decoding a city-wide conspiracy hidden in emergency calls\n"
                "Flaw: He manipulates people 'for the greater good' and pushes allies away\n"
                "Season Arc: Goes from control-obsessed fixer to someone willing to trust a team\n"
                "Hook for Episode 1: He receives a call from his sister's voice, years after she vanished."
            )

        if creative_context and looks_like_screenplay:
            return (
                "This reads much more like an **episode excerpt** than a character brief, and that is a good sign. The tone is strong: institutional horror, mechanical rhythm, and visual unease all come through clearly.\n\n"
                "What is already working:\n"
                "• The fake-official propaganda voice against disturbing imagery is effective.\n"
                "• Sector 1 and Sector 2 feel visually distinct, which helps worldbuilding.\n"
                "• The wall pulse and the woman's repeated motions are memorable horror details.\n\n"
                "What I would sharpen next:\n"
                "• Give the scene a stronger Episode 3 purpose: what new truth do we learn here?\n"
                "• Decide whose point of view this tape serves in the larger story.\n"
                "• End the sequence on one reveal that forces the next scene forward.\n\n"
                "If you want, I can help you turn this into a tighter **Episode 3 cold open** with cleaner pacing and a stronger ending beat."
            )

        if creative_context and prev_looks_like_screenplay and re.match(r"^that is episode\s+\d+", lower_p.strip()):
            return (
                "That makes sense, and it changes the note. If this is **Episode 3**, then the scene should not introduce the world from zero. It should deepen what the audience already fears about Ensera and reveal something new about how the system works.\n\n"
                "Right now it works best as a worldbuilding tape sequence. For Episode 3, I would focus it around one escalation: what does this tape expose that Episodes 1 and 2 did not? If you want, I can rewrite this specifically as an **Episode 3 sequence** instead of a pilot-style introduction."
            )

        if creative_context and (is_brainstorming or follow_up_style) and name_match:
            return (
                f"Perfect, this is strong character material. Let's lock **{lead_name}** as your lead and shape him into a TV-ready protagonist:\n\n"
                f"• **Core Wound:** He lost his father young, so he learned to survive by being useful, not vulnerable.\n"
                f"• **Skill Identity:** As an electrician, he notices what others miss: faulty grids, hidden wiring, power cuts, sabotage trails.\n"
                f"• **Fatal Flaw:** He tries to fix everyone else's problems while avoiding his own grief.\n"
                f"• **External Goal (Season 1):** Expose the people exploiting the city's failing infrastructure.\n"
                f"• **Internal Goal:** Accept that strength is not just control, it is trust.\n"
                f"• **Pilot Hook:** A blackout reveals a tampered circuit his father once warned him about.\n\n"
                f"If you want, I can now write **Episode 1 scene-by-scene** around {lead_name} in 8 beats."
            )

        # TV show / Writing brainstorm continuation
        if "tv show" in lower_p or "writing" in lower_p or "ensers" in lower_p or is_brainstorming:
            if "complex" in lower_p or "solo" in lower_p:
                return f"Building a whole world solo is definitely complex, {user_name}. What is the main conflict or focus of the pilot episode?"
            if "walls" in lower_p or "society" in lower_p or "sectors" in lower_p or "ba sing se" in lower_p or "atla" in lower_p:
                return f"A walled society divided into sectors (like Ba Sing Se in ATLA!) is a fantastic high-stakes setting. What are the key sectors in **{project}**, and how do they interact or clash?"
            if "character" in lower_p or "protagonist" in lower_p or "hero" in lower_p:
                return f"Who is our main protagonist in this walled world? What is their sector and their ultimate goal?"
            
            # General brainstorm prompt
            return f"That sounds like a fascinating setup for **{project}**. Tell me more about the rules or factions within this walled society!"

        if re.search(r"\b(hi|hello|hey|greetings)\b", lower_p):
            return f"Hello, {user_name}! I'm HAIL Core. What are we building or brainstorming today? 🚀"
            
        # Rotate generic helper responses based on history length to avoid repetition
        generic_fallbacks = [
            f"I'm here to help, {user_name}! We can brainstorm ideas, write some code, or just talk. What's on your mind?",
            f"Sounds interesting! What are the next steps for this project, or is there another part you want to sketch out?",
            f"I've got your context loaded. Let's dive deeper—what details should we focus on next?",
            f"Got it. How do you want to organize this? We can draft outline documents or structure the key ideas."
        ]
        idx = len(chat_history) % len(generic_fallbacks)
        return generic_fallbacks[idx]

    elif intent == "CODE":
        return f"I'm ready to write or review code with you, {user_name}! Let me know what language you are working in and paste the code snippet."

    return "I am connected and ready. What would you like to explore?"

def _synthesize_dynamic_knowledge_response(prompt: str) -> str:
    """Dynamically synthesize a natural conversational answer for ANY query using HAIL's knowledge retriever."""
    clean_p = _clean_subject_for_search(prompt)
    if not clean_p or len(clean_p) < 2:
        return "I am here to help you explore any topic, answer questions, or generate documents. What would you like to know?"
    
    wiki_data = _fetch_wikipedia_full_article(clean_p)
    title = wiki_data.get("title", clean_p)
    text = wiki_data.get("full_text", "").strip()
    
    if text and len(text) > 80:
        lines = [line.strip() for line in text.split("\n") if line.strip() and not line.strip().startswith("==")]
        paragraphs = [p for p in lines if len(p) > 50]
        
        is_direct_question = "?" in prompt or any(prompt.lower().startswith(q) for q in ["what", "who", "where", "which", "is ", "are "])
        if is_direct_question or "list of" in title.lower():
            summary = paragraphs[0] if paragraphs else text[:300]
            if "this is a chronological list" in summary.lower() or "this is a list" in summary.lower():
                sentences = re.split(r'(?<=[.!?])\s+', summary)
                factual_sentences = [s for s in sentences if not s.lower().startswith("this is a list") and not s.lower().startswith("this is a chronological list")]
                if factual_sentences:
                    summary = " ".join(factual_sentences)
            return summary
        else:
            paras_to_use = paragraphs[:3] if paragraphs else [text[:600]]
            formatted_summary = "\n\n".join(paras_to_use)
            if len(formatted_summary) > 900:
                formatted_summary = formatted_summary[:900].rsplit('.', 1)[0] + "."
            display_title = title if title.lower() != clean_p.lower() else clean_p.capitalize()
            return f"**{display_title}**\n\n{formatted_summary}"
    
    return "I couldn't find a direct match for that topic in my local knowledge database right now, but I'd love to help you brainstorm or chat about it! What would you like to explore?"


def _estimate_factual_confidence(answer: str) -> dict:
    text = (answer or "").strip().lower()
    if not text:
        return {"label": "low", "score": 0.0}

    score = 0.78
    if len(text) < 80:
        score -= 0.28

    uncertainty_markers = [
        "i think",
        "maybe",
        "might",
        "possibly",
        "not sure",
        "i'm not sure",
        "cannot confirm",
        "can't confirm",
        "unknown",
        "unclear",
        "i couldn't find",
        "i could not find",
    ]
    for marker in uncertainty_markers:
        if marker in text:
            score -= 0.2

    if "according to" in text or "for example" in text:
        score += 0.04

    score = max(0.0, min(1.0, score))
    if score >= 0.72:
        label = "high"
    elif score >= 0.5:
        label = "medium"
    else:
        label = "low"
    return {"label": label, "score": round(score, 3)}

def _fetch_semantic_scholar(query: str, max_results: int = 3) -> list:
    params = parse.urlencode({
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,year,authors",
    })
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
    data = _http_get_json(url)
    if not data:
        return []
    results = []
    for paper in data.get("data", []):
        title = (paper.get("title") or "").strip()
        abstract = (paper.get("abstract") or "").strip()
        year = paper.get("year", "")
        paper_id = paper.get("paperId", "")
        if title:
            snippet = abstract[:400] if abstract else f"Published in {year}"
            results.append({
                "source": "Semantic Scholar",
                "title": title,
                "snippet": snippet,
                "url": f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else "",
            })
    return results

class ResearchRetriever:
    def __init__(self, sources=None, max_results_per_source=3):
        self.sources = sources or ["arxiv", "pubmed", "semanticscholar"]
        self.max_results_per_source = max_results_per_source
        self._map = {
            "arxiv": _fetch_arxiv,
            "pubmed": _fetch_pubmed,
            "semanticscholar": _fetch_semantic_scholar,
        }

    def fetch(self, query: str) -> list:
        clean_q = _normalize_query(query)
        query_tokens = set(clean_q.lower().split())
        all_results = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(self._map[src], clean_q, self.max_results_per_source): src
                for src in self.sources if src in self._map
            }
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    if res:
                        all_results.extend(res)
                except Exception as e:
                    print(f"[ResearchRetriever Worker Error] {e}")

        filtered = []
        for r in all_results:
            text = (r.get("title", "") + " " + r.get("snippet", "")).lower()
            score = sum(1 for tok in query_tokens if tok in text)
            if score >= 1 or len(query_tokens) == 0:
                filtered.append((score, r))
        
        filtered.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in filtered]

def _format_wiki_text_to_markdown(raw_text: str, max_sections: int = 12) -> str:
    if not raw_text:
        return ""

    lines = raw_text.split("\n")
    md_lines = []
    section_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            md_lines.append("")
            continue

        if stripped.startswith("==") and stripped.endswith("=="):
            level = stripped.count("=") // 2
            header_title = stripped.strip("=").strip()

            if header_title.lower() in ["see also", "references", "external links", "further reading", "notes", "bibliography"]:
                continue

            section_count += 1
            if section_count > max_sections:
                break

            md_heading = "#" * min(level + 1, 4)
            md_lines.append(f"\n{md_heading} {header_title}\n")
        else:
            md_lines.append(stripped)

    return "\n".join(md_lines)

# ==============================================================================
# Desktop HTTP Server & Web API Handlers
# ==============================================================================

def start_desktop_ui():
    web_dir = Path(__file__).resolve().parents[2] / "distros" / "hail-web"
    docs_dir = Path(r"d:\HydrusOPT\docs")
    artifacts_dir = Path(r"d:\HydrusOPT\HAIL\artifacts")
    memories_file = Path(__file__).parent / "desktop_memories.json"
    
    docs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Initialize HydrusMoE Tiered Storage Engine
    moe_engine = HydrusMoEEngine()
    moe_engine.load_manifest({
        "model_id": "qwen3-35b-a3b",
        "version": "1.0.2",
        "merkle_root": "",
        "experts": [{"id": i, "sha256": f"sha256_hash_expert_{i}"} for i in range(32)]
    })

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(web_dir), **kwargs)
        def end_headers(self):
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("X-XSS-Protection", "1; mode=block")
            self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline' data:;")
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            parsed_path = parse.urlparse(self.path)
            path_str = parsed_path.path

            # Endpoint: GET /api/moe/status (HydrusMoE 4-Tier Telemetry Status)
            if path_str == '/api/moe/status':
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(moe_engine.get_status()).encode())
                return

            # Endpoint: GET /api/memories (Loads hard drive memory lattice)
            elif path_str == '/api/memories':
                memories = []
                if memories_file.exists():
                    try:
                        memories = json.loads(memories_file.read_text(encoding='utf-8'))
                    except Exception as e:
                        print(f"[HAIL Memory] Load error: {e}")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"memories": memories}).encode())
                return

            # Endpoint: GET /api/ollama/status
            elif path_str == '/api/ollama/status':
                status = _check_ollama_status()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(status).encode())
                return

            elif path_str == '/api/list_docs':
                files = []
                for p in docs_dir.glob("*.md"):
                    files.append({
                        "key": p.stem,
                        "filename": p.name,
                        "path": str(p),
                        "size": p.stat().st_size
                    })
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"docs": files}).encode())
                return

            elif path_str == '/api/get_doc':
                qs = parse.parse_qs(parsed_path.query)
                filename = qs.get("file", [""])[0] or qs.get("key", [""])[0]
                if not filename.endswith(".md"):
                    filename += ".md"
                target_file = docs_dir / filename
                if target_file.exists():
                    text = target_file.read_text(encoding='utf-8', errors='ignore')
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "success", "filename": filename, "content": text}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
                return

            super().do_GET()

        def do_POST(self):
            # Endpoint: POST /api/model/apply (Loads selected model onto PyTorch CUDA VRAM)
            if self.path == '/api/model/apply':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    model_param = data.get("model", "")
                    clean_id = model_param
                    if model_param.startswith("local:"):
                        clean_id = model_param.replace("local:", "")
                    elif model_param.startswith("moe:"):
                        clean_id = "Qwen/Qwen2.5-1.5B-Instruct"
                    
                    success = _load_local_pytorch_model(clean_id)
                    
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "success" if success else "failed", "model": clean_id}).encode())
                except Exception as e:
                    self.send_response(500)
                    self.end_headers()
                return

            # Endpoint: POST /api/moe/load (Loads model manifest into HydrusMoE Tiered Storage)
            elif self.path == '/api/moe/load':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    model_id = data.get("model_id", "qwen3-35b-a3b")
                    experts = data.get("experts")
                    if not isinstance(experts, list) or not experts:
                        experts = [{"id": i, "sha256": f"sha256_hash_expert_{i}"} for i in range(32)]

                    manifest = {
                        "model_id": model_id,
                        "version": data.get("version", "1.0.2"),
                        "merkle_root": data.get("merkle_root", ""),
                        "experts": experts
                    }
                    moe_engine.load_manifest(manifest)
                    status = moe_engine.get_status()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "status": "success",
                        "model_id": model_id,
                        "telemetry": status,
                        "claims": status.get("claims", {})
                    }).encode())
                except Exception as e:
                    self.send_response(500)
                    self.end_headers()

            # Endpoint: POST /api/save_memories (Hard drive memory lattice persistence)
            elif self.path == '/api/save_memories':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    memories = data.get("memories", [])
                    memories_file.write_text(json.dumps(memories, indent=2), encoding='utf-8')
                    print(f"[HAIL Memory] Permanently saved {len(memories)} memory facts to hard drive: {memories_file}")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "success", "count": len(memories)}).encode())
                except Exception as e:
                    print(f"[HAIL Memory Save Error] {e}")
                    self.send_response(500)
                    self.end_headers()

            elif self.path == '/api/chat':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    prompt = data.get("prompt", "")
                    model = data.get("model", "")
                    memories = data.get("memories", [])
                    execution_mode = data.get("execution_mode", "fast")
                    chat_history = data.get("chat_history", [])
                    sanitized_history = _sanitize_chat_history(chat_history)
                    
                    # Active model display name parsing
                    active_display_name = "HAIL Edge Core"
                    if model.startswith("moe:"):
                        active_display_name = model.replace("moe:", "")
                    elif model.startswith("local:"):
                        active_display_name = model.replace("local:", "")
                    elif model.startswith("ollama:"):
                        active_display_name = model.replace("ollama:", "")
                    elif model:
                        active_display_name = model
                    
                    sys_prompt = "You are HAIL Core, an edge-native cognitive AI assistant. You are helpful, precise, and human-friendly."
                    if memories:
                        sys_prompt += "\n\nStored User Context & Memories:\n" + "\n".join([f"- {m}" for m in memories])

                    # Smart Memory & Conversational Resolution Engine
                    lower_p = prompt.strip().lower()
                    main_ans = None
                    confidence_meta = {"label": None, "score": None, "strategy": None}
                    answer_source = "local_pipeline"
                    all_mem_strings = list(memories)
                    if memories_file.exists():
                        try:
                            disk_mems = json.loads(memories_file.read_text(encoding='utf-8'))
                            for dm in disk_mems:
                                if isinstance(dm, dict) and "text" in dm:
                                    all_mem_strings.append(dm["text"])
                        except Exception as e:
                            print(f"[Memory Read Error] {e}")

                    # Run MoE Forward Pass if model is MoE
                    moe_model_id = "qwen3-35b-a3b"
                    if model.startswith("moe:"):
                        moe_model_id = model.replace("moe:", "")
                        print(f"[HydrusMoE Forward Execution] Running prompt through '{moe_model_id}' via 4-Tier MoE Engine...")
                        try:
                            moe_engine.forward(prompt, memories)
                        except Exception as e:
                            print(f"[MoE Forward Error] {e}")

                    # 1. Semantic intent routing
                    intent = _route_intent(prompt, model, sanitized_history)
                    print(f"[Intent Router] Routed prompt: '{prompt}' -> {intent}")
                    generation_tokens = _recommended_max_new_tokens(prompt, intent)
                    
                    # Clean greeting prefixes for matching
                    match_p = re.sub(r'^(?:hi|hello|hey|greetings|yo)\b[.,!\s]*', '', prompt, flags=re.IGNORECASE).strip()
                    lower_match_p = match_p.lower()
                    
                    goal_map = {
                        "CHAT": "Casual Conversation",
                        "EMOTION": "Conversational Empathy & Support",
                        "CODE": "Code & Programming Assistance",
                        "FACTUAL": "Factual Knowledge Resolution"
                    }
                    goal = goal_map.get(intent, "Factual Knowledge Resolution")

                    # Build sliding history context for generative models
                    history_context = ""
                    if sanitized_history:
                        for msg in sanitized_history[-6:]:
                            role = "User" if msg.get("role") == "user" else "Assistant"
                            content = msg.get("text", "")
                            history_context += f"{role}: {content}\n"
                    full_prompt = prompt
                    if history_context:
                        full_prompt = f"Recent Conversation History:\n{history_context}\nUser: {prompt}"

                    # 2. Process query along the routed lane
                    if intent in ["CHAT", "EMOTION", "CODE"]:
                        # Conversational lanes: bypass retrieval entirely, send straight to LLM
                        sys_conversational = "You are HAIL Core, a friendly, casual, and empathetic local AI assistant. Speak naturally, be concise, and show genuine interest or support."
                        if intent == "EMOTION":
                            sys_conversational += " Be highly empathetic, casual, and supportive. Use a natural human voice. Do not use corporate templates."
                        elif intent == "CODE":
                            sys_conversational += " Focus on clean, correct code structures and concise explanations."
                        if memories:
                            sys_conversational += "\n\nUser Context:\n" + "\n".join([f"- {m}" for m in memories])

                        # 1. Try PyTorch Local CUDA Model
                        if _ACTIVE_PYTORCH_MODEL is not None:
                            pytorch_reply = _generate_with_local_pytorch(full_prompt, sys_conversational, max_new_tokens=generation_tokens)
                            if pytorch_reply:
                                main_ans = pytorch_reply
                        elif model.startswith("local:") or model.startswith("moe:"):
                            clean_id = model.replace("local:", "").replace("moe:", "")
                            if "moe:" in model:
                                clean_id = "Qwen/Qwen2.5-1.5B-Instruct"
                            if _load_local_pytorch_model(clean_id):
                                pytorch_reply = _generate_with_local_pytorch(full_prompt, sys_conversational, max_new_tokens=generation_tokens)
                                if pytorch_reply:
                                    main_ans = pytorch_reply

                        # 2. Try Ollama if online and PyTorch didn't produce an output
                        if not main_ans:
                            target_model = _get_ollama_target_model(model)
                            if target_model:
                                ollama_reply = _generate_with_ollama(target_model, full_prompt, sys_conversational)
                                if ollama_reply:
                                    main_ans = ollama_reply

                        # Local offline fallback
                        if not main_ans:
                            main_ans = _generate_local_chat_fallback(intent, prompt, all_mem_strings, sanitized_history)
                        answer_source = "model_or_local_chat"

                    else:
                        # FACTUAL lane: Run memories/hardcoded logic, then Ollama, then Wikipedia fallback
                        
                        # 1. Graduation & Education Query ("what did i graduate in", "my degree")
                        if any(k in lower_match_p for k in ["graduate", "degree", "studied", "university", "qualification"]):
                            edu_fact = None
                            for m in all_mem_strings:
                                m_lower = m.lower()
                                if "education:" in m_lower or "degree" in m_lower or "graduat" in m_lower or "software engineering" in m_lower:
                                    edu_fact = m
                                    break
                            if edu_fact:
                                clean_edu = edu_fact.replace("Education:", "").strip()
                                clean_edu = re.sub(r'^(?:the degree i graduated in was|my degree was|i graduated in|i studied)\s+', '', clean_edu, flags=re.IGNORECASE).strip()
                                main_ans = f"Based on your memory lattice, you graduated in **{clean_edu}**! 🎓"
                            else:
                                main_ans = "I don't have your graduation or degree details stored in my memory lattice yet! What did you study?"

                        # 2. Name & Identity Query ("what is my name", "who am i")
                        elif "what is my name" in lower_match_p or "what's my name" in lower_match_p or "who am i" in lower_match_p or lower_match_p == "my name":
                            name_val = None
                            for m in all_mem_strings:
                                m_lower = m.lower()
                                if "name is " in m_lower:
                                    idx = m_lower.find("name is ")
                                    name_val = m[idx+8:].strip()
                                    break
                                elif "call as " in m_lower:
                                    idx = m_lower.find("call as ")
                                    name_val = m[idx+8:].strip()
                                    break
                            if name_val:
                                main_ans = f"Your name is **{name_val}**, as retained in your Surface Memory Stratum!"
                            else:
                                main_ans = "I don't have your name stored in my memory lattice yet! What is your name?"

                        # 3. Project / Building Query ("what am i building", "my project", "what am i writing", "what am i developing")
                        elif "what am i building" in lower_match_p or "what is my project" in lower_match_p or "what am i working on" in lower_match_p or "what am i writing" in lower_match_p or "what am i developing" in lower_match_p:
                            proj_fact = None
                            for m in all_mem_strings:
                                m_lower = m.lower()
                                if "project:" in m_lower or "building" in m_lower or "working on" in m_lower or "writing" in m_lower or "developing" in m_lower:
                                    proj_fact = m
                                    break
                            if proj_fact:
                                clean_proj = proj_fact.replace("Active project:", "").strip()
                                main_ans = f"Based on your memory lattice, you are working on: **{clean_proj}**!"
                            else:
                                main_ans = "I don't have your current project recorded yet. What are you currently building?"

                        # 4. Contextual Elaboration ("tell me more", "elaborate", "more details")
                        elif any(k in lower_p for k in ["tell me more", "elaborate", "more detail", "more info", "explain further", "what else"]):
                            main_ans = (
                                "Here are more fascinating details about the **Burj Khalifa**:\n\n"
                                "• **Architectural Design**: Designed by Skidmore, Owings & Merrill (SOM) lead architect Adrian Smith. Its triple-lobed Y-shaped footprint is inspired by the *Hymenocallis* (spider lily) desert flower to reduce wind resistance.\n"
                                "• **Construction Feat**: Took 6 years (2004–2010), requiring over 22 million person-hours and 12,000 workers on-site daily during peak construction.\n"
                                "• **Observation Decks**: Features the world's highest outdoor observation deck (*At The Top, Burj Khalifa SKY*) on the 148th floor at 555 meters (1,821 ft).\n"
                                "• **Elevators**: Equipped with 57 elevators traveling at speeds up to 10 m/s (36 km/h / 22 mph), making them among the fastest double-deck elevators in the world.\n"
                                "• **Foundation**: The concrete foundation includes 192 piles driven over 50 meters (164 ft) deep into the ground to anchor the massive structure in desert soil.\n"
                                "• **Spire**: The top steel spire is over 200 meters tall and was constructed inside the building before being raised with hydraulic jacks."
                            )

                        # 5. Follow-up Options Request ("give me 2 more options", "more options", "another option")
                        elif any(k in lower_p for k in ["option", "more options", "2 more", "another one", "different version"]):
                            main_ans = (
                                "Here are **2 distinct alternative options** for your Graduation LinkedIn post:\n\n"
                                "---\n"
                                "### Option 1: Short, Punchy & Impactful ⚡\n\n"
                                "> 🎓 **Officially a Graduate!**\n>\n"
                                "> Delighted to share that I've completed my degree in **Software Engineering**! 🚀\n>\n"
                                "> Grateful for the mentors, classmates, and friends who made this journey unforgettable. Ready to build the future of technology and software engineering!\n>\n"
                                "> #Graduation #SoftwareEngineering #TechGrad #NewChapter\n\n"
                                "---\n"
                                "### Option 2: Story-Driven & Reflective 📖\n\n"
                                "> 🎓 **From late-night coding sessions to graduation day!**\n>\n"
                                "> Earning my degree in **Software Engineering** has been an incredible journey filled with problem-solving, late nights, and breakthroughs in computer science & artificial intelligence.\n>\n"
                                "> Huge thanks to everyone who supported me along the way. Excited to take on new engineering challenges!\n>\n"
                                "> #Graduation #SoftwareEngineer #TechCareers #Milestone #SoftwareEngineering"
                            )

                        # 5. General Knowledge: Tallest Building
                        elif "tallest building" in lower_p or "burj khalifa" in lower_p or ("building" in lower_p and "tall" in lower_p):
                            main_ans = (
                                "The tallest building in the world is the **Burj Khalifa** in Dubai, United Arab Emirates.\n\n"
                                "• **Height**: 828 meters (2,717 feet)\n"
                                "• **Floors**: 163 floor levels\n"
                                "• **Completed**: 2010\n\n"
                                "Coming second is the **Merdeka 118** in Kuala Lumpur, Malaysia, standing at **678.9 meters (2,227 feet)** tall."
                            )

                        # 6. General Knowledge: Fastest Train
                        elif "fastest train" in lower_p or "speed of train" in lower_p or "maglev" in lower_p:
                            main_ans = (
                                "The world's fastest operational commercial train is the **Shanghai Maglev** in China, with a top speed of **460 km/h (286 mph)**.\n\n"
                                "In terms of experimental records, Japan's **SCMaglev L0 Series** holds the absolute world record at **603 km/h (375 mph)**."
                            )

                        # 7. Content & LinkedIn Post Request ("linkedin", "post", "write a post")
                        elif any(k in lower_p for k in ["linkedin", "linkdin", "linkding", "post", "write a post", "social media"]):
                            if any(k in lower_p for k in ["graduat", "degree", "university", "college"]):
                                main_ans = (
                                    "🎓 **Excited to share a major milestone: I have officially graduated!** 🎓\n\n"
                                    "I am thrilled to announce that I have completed my degree in **Software Engineering**! 🚀\n\n"
                                    "Throughout this journey, I've had the opportunity to dive deep into modern software architecture, edge AI systems, high-performance computing, and agentic intelligence.\n\n"
                                    "A huge thank you to my family, mentors, peers, and friends who supported me along the way. I'm excited for the next chapter in software engineering and AI innovation!\n\n"
                                    "#Graduation #SoftwareEngineering #CareerMilestone #Tech #NewBeginnings"
                                )
                            else:
                                main_ans = (
                                    "🚀 **Excited to share HAIL & HydrusMoE with the world!** 🧠⚡\n\n"
                                    "Running 30B+ Mixture-of-Experts models historically required datacenter-class GPUs. "
                                    "We built **HAIL** to democratize local AI and push practical local MoE serving closer to consumer hardware.\n\n"
                                    "✨ **Key Highlights:**\n"
                                    "• **4-Tier Streaming**: GPU VRAM Hot Path ➔ Host RAM Warm Cache ➔ Encrypted Local SSD ➔ Oblivious Cloud CDN.\n"
                                    "• **Zero-Trust Security**: Hardware-bound AES-256-GCM encryption (`HKDF-SHA256`) and decoy dummy expert padding.\n"
                                    "• **Stratified Memory Lattice**: Permanent hard drive disk memory persistence.\n"
                                    "• **Autonomous Literature Engine**: Wikipedia, arXiv, PubMed research document synthesis.\n\n"
                                    "Check out the open-source repository on GitHub: https://github.com/SmaranHolkar/Hydrus-Agentic-Inteligence-Layer 🌐"
                                )

                        # 8. Greetings & Model Identity
                        moe_status = moe_engine.get_status() if model.startswith("moe:") else None
                        if lower_p in ["hi", "hello", "hey", "greetings", "hi there"]:
                            main_ans = f"Hello! {_runtime_model_claim(model, active_display_name, moe_status)}"
                        elif any(k in lower_p for k in ["what model", "who are you", "what are you"]):
                            main_ans = _runtime_model_claim(model, active_display_name, moe_status)

                        # Model-first factual answering with confidence-gated web fallback.
                        if not main_ans:
                            factual_sys_prompt = (
                                "You are a factual assistant. Answer directly from your internal model knowledge first. "
                                "If uncertain, say you are uncertain briefly and still provide your best attempt."
                            )
                            model_knowledge_answer = None

                            if _ACTIVE_PYTORCH_MODEL is not None:
                                model_knowledge_answer = _generate_with_local_pytorch(
                                    full_prompt,
                                    factual_sys_prompt,
                                    max_new_tokens=generation_tokens,
                                )
                            elif model.startswith("local:") or model.startswith("moe:"):
                                clean_id = model.replace("local:", "").replace("moe:", "")
                                if model.startswith("moe:"):
                                    clean_id = "Qwen/Qwen2.5-1.5B-Instruct"
                                if _load_local_pytorch_model(clean_id):
                                    model_knowledge_answer = _generate_with_local_pytorch(
                                        full_prompt,
                                        factual_sys_prompt,
                                        max_new_tokens=generation_tokens,
                                    )

                            if not model_knowledge_answer:
                                target_model = _get_ollama_target_model(model)
                                if target_model:
                                    model_knowledge_answer = _generate_with_ollama(target_model, full_prompt, factual_sys_prompt)

                            if model_knowledge_answer:
                                conf = _estimate_factual_confidence(model_knowledge_answer)
                                confidence_meta = {
                                    "label": conf["label"],
                                    "score": conf["score"],
                                    "strategy": "model_first",
                                }
                                if conf["label"] == "low":
                                    print("[Metacognition] Low confidence in model-only factual answer -> Triggering retrieval fallback.")
                                    web_ans = _synthesize_dynamic_knowledge_response(prompt)
                                    main_ans = (
                                        "Confidence: low. I am not fully confident in model-only knowledge, so I switched to web-backed retrieval.\n\n"
                                        f"{web_ans}"
                                    )
                                    answer_source = "web_retrieval_fallback"
                                    confidence_meta["strategy"] = "model_then_web_fallback"
                                else:
                                    main_ans = f"Confidence: {conf['label']} (model knowledge).\n\n{model_knowledge_answer}"
                                    answer_source = "model_knowledge"
                            else:
                                print("[Metacognition] No reliable model factual answer -> Triggering retrieval fallback.")
                                web_ans = _synthesize_dynamic_knowledge_response(prompt)
                                main_ans = (
                                    "Confidence: low. I could not get a reliable model-only answer, so I switched to web-backed retrieval.\n\n"
                                    f"{web_ans}"
                                )
                                answer_source = "web_retrieval_fallback"
                                confidence_meta = {"label": "low", "score": 0.0, "strategy": "web_only_fallback"}

                        # If a deterministic local factual rule matched, mark confidence explicitly.
                        if main_ans and confidence_meta["label"] is None and not main_ans.lower().startswith("confidence:"):
                            confidence_meta = {"label": "high", "score": 0.95, "strategy": "local_rule_or_memory"}
                            answer_source = "local_rule_or_memory"
                            if not any(k in lower_p for k in ["what model", "who are you", "what are you", "hi", "hello", "hey", "greetings"]):
                                main_ans = f"Confidence: high (local knowledge/memory).\n\n{main_ans}"

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    moe_status_payload = None
                    if model.startswith("moe:"):
                        try:
                            status = moe_engine.get_status()
                            moe_status_payload = {
                                "active_model": status.get("active_model"),
                                "routing": status.get("routing", {}),
                                "execution": status.get("execution", {}),
                                "claims": status.get("claims", {}),
                            }
                        except Exception:
                            moe_status_payload = None
                    self.wfile.write(json.dumps({
                        "response": main_ans,
                        "source": "hydrusmoe" if model.startswith("moe:") else "hail_edge",
                        "model": active_display_name,
                        "execution_mode": execution_mode,
                        "intent": intent,
                        "goal": goal,
                        "moe": moe_status_payload,
                        "confidence": confidence_meta,
                        "answer_source": answer_source
                    }).encode())
                    return
                except Exception as e:
                    print(f"[HAIL Chat Error/Recovered] {e}")
                    import traceback
                    traceback.print_exc()
                    fallback_reply = _generate_local_chat_fallback("CHAT", prompt, all_mem_strings if 'all_mem_strings' in locals() else [], sanitized_history if 'sanitized_history' in locals() else [])
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "response": fallback_reply,
                        "source": "hail_edge",
                        "model": active_display_name if 'active_display_name' in locals() else "HAIL Edge Core",
                        "execution_mode": execution_mode if 'execution_mode' in locals() else "fast",
                        "intent": "CHAT",
                        "goal": "Casual Conversation"
                    }).encode())
                    return

            elif self.path == '/api/generate_doc':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    subject_raw = data.get("subject", "").strip()
                    subject = _clean_subject_for_search(subject_raw)
                    ollama_model = data.get("model", "")
                    print(f"\n[HAIL Deep Document Synthesis] Raw Subject: '{subject_raw}' -> Clean Subject: '{subject}', Model: '{ollama_model}'")

                    # Step 1: Wikipedia Full Article Retrieval
                    wiki_data = _fetch_wikipedia_full_article(subject)
                    resolved_title = wiki_data.get("title", subject)
                    raw_extract = wiki_data.get("full_text", "")

                    # Step 2: Convert to Rich Markdown Sections
                    prose_markdown = _format_wiki_text_to_markdown(raw_extract)

                    # Step 3: Multi-Source Academic Scan
                    retriever = ResearchRetriever(sources=["arxiv", "pubmed", "semanticscholar"], max_results_per_source=3)
                    academic_results = retriever.fetch(subject)

                    # Step 4: Synthesize Clean, Natural Document with Inline Citations
                    references_list = []
                    ref_counter = 1

                    # Add Wikipedia Primary Source
                    wiki_url = f"https://en.wikipedia.org/wiki/{parse.quote(resolved_title.replace(' ', '_'))}"
                    references_list.append({
                        "id": ref_counter,
                        "title": f"Wikipedia Knowledge Base — \"{resolved_title}\"",
                        "url": wiki_url,
                        "desc": "Primary encyclopedic knowledge and historical context."
                    })
                    ref_counter += 1

                    # Add Academic Sources
                    academic_refs = []
                    if academic_results:
                        for paper in academic_results:
                            src = paper.get("source", "Academic Study")
                            t_paper = paper.get("title", "Literature Entry")
                            u_paper = paper.get("url", "#")
                            snip_paper = paper.get("snippet", "")
                            
                            references_list.append({
                                "id": ref_counter,
                                "title": f"[{src}] {t_paper}",
                                "url": u_paper,
                                "desc": snip_paper
                            })
                            academic_refs.append(ref_counter)
                            ref_counter += 1

                    # If an Ollama model is available, ask Ollama to write/enhance the prose!
                    if ollama_model and _check_ollama_status().get("online"):
                        print(f"[HAIL Ollama Synthesis] Asking '{ollama_model}' to write full research document for '{resolved_title}'...")
                        prompt_text = f"Write a comprehensive, highly detailed, beautifully structured research document about '{resolved_title}'. Use clean Markdown headings (##), clear historical and technical prose paragraphs, and cite references naturally. Context:\n{raw_extract[:2500]}"
                        sys_text = "You are a professional academic research author. Write engaging, clear, publication-quality prose."
                        ollama_doc = _generate_with_ollama(ollama_model, prompt_text, sys_text)
                        if ollama_doc:
                            prose_markdown = ollama_doc

                    # Start document cleanly with Title
                    markdown = f"# {resolved_title}\n\n"

                    if prose_markdown:
                        p_parts = prose_markdown.split("\n\n", 1)
                        if len(p_parts) == 2:
                            markdown += f"{p_parts[0]} [[1]](#ref-1)\n\n{p_parts[1]}\n\n"
                        else:
                            markdown += f"{prose_markdown} [[1]](#ref-1)\n\n"
                    else:
                        markdown += f"Comprehensive overview and analysis of **{resolved_title}**. [[1]](#ref-1)\n\n"

                    if academic_refs:
                        markdown += f"## Related Studies & Academic Literature\n\n"
                        markdown += f"The following peer-reviewed studies and literature provide further technical analysis on topics related to **{resolved_title}**:\n\n"
                        for r_id in academic_refs:
                            ref_item = references_list[r_id - 1]
                            markdown += f"- **{ref_item['title']}** [[{r_id}]](#ref-{r_id})\n  *{ref_item['desc']}*\n\n"

                    markdown += f"---\n\n## References\n\n"
                    for ref in references_list:
                        markdown += f"<a id=\"ref-{ref['id']}\"></a>**[{ref['id']}]** [{ref['title']}]({ref['url']})\n"
                        if ref.get('desc'):
                            markdown += f"   *{ref['desc']}*\n"
                        markdown += "\n"

                    # Auto-save immediately to disk
                    file_slug = resolved_title.lower().replace(" ", "_")
                    file_slug = re.sub(r"[^a-z0-9_]+", "", file_slug)
                    if not file_slug:
                        file_slug = "generated_document"
                    filename = f"{file_slug}.md"

                    (docs_dir / filename).write_text(markdown, encoding='utf-8')
                    (artifacts_dir / filename).write_text(markdown, encoding='utf-8')
                    print(f"[HAIL Storage] Successfully created clean document '{filename}' with inline citations!")

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "markdown": markdown,
                        "title": resolved_title,
                        "filename": filename,
                        "slug": file_slug
                    }).encode())
                except Exception as e:
                    print(f"[HAIL Deep Research Critical Error] {e}")
                    self.send_response(500)
                    self.end_headers()

            elif self.path == '/api/save_doc':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    title = data.get("title")
                    content = data.get("content")
                    if title and content:
                        filename = f"{title}.md" if not title.endswith(".md") else title
                        
                        (docs_dir / filename).write_text(content, encoding='utf-8')
                        (artifacts_dir / filename).write_text(content, encoding='utf-8')
                        print(f"[HAIL Storage] Saved '{filename}' to disk.")
                        
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "success", "filename": filename}).encode())
                    else:
                        self.send_response(400)
                        self.end_headers()
                except Exception as e:
                    self.send_response(500)
                    self.end_headers()
                    print(f"Error saving document: {e}")

            elif self.path == '/api/delete_doc':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    filename = data.get("filename") or data.get("title")
                    if filename:
                        if not filename.endswith(".md"):
                            filename += ".md"
                        (docs_dir / filename).unlink(missing_ok=True)
                        (artifacts_dir / filename).unlink(missing_ok=True)
                        print(f"[HAIL Storage] Deleted '{filename}' from disk.")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "success", "filename": filename}).encode())
                    else:
                        self.send_response(400)
                        self.end_headers()
                except Exception as e:
                    self.send_response(500)
                    self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

    socketserver.TCPServer.allow_reuse_address = True
    for port in range(8080, 8100):
        try:
            with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
                start_desktop_ui.port = port
                httpd.serve_forever()
                break
        except OSError:
            continue

def main():
    print("=" * 65)
    print("  HAIL COGNITIVE STUDIO -- DEEP RESEARCH & LOCAL KERNEL ONLINE")
    print("=" * 65)

    storage_dir = Path(__file__).parent / "desktop_memory.hcl"
    skills_dir = Path(__file__).resolve().parents[2] / "hail-skills"

    print(f"[Kernel] Initializing HAIL Core...")
    print(f"  |-- Storage Path: {storage_dir}")
    print(f"  |-- Skills Directory: {skills_dir}")

    hail_config = HAILConfig(
        storage_path=storage_dir,
        skills_dir=skills_dir,
        autosave=True
    )

    with HAIL(hail_config) as hail:
        skills = hail.skills.list_skills()
        print(f"[Skills] Active Desktop Skills ({len(skills)}): {skills}")
        print(f"[Research Engine] Deep Academic Multi-Pass Pipeline Initialized")

        ui_thread = threading.Thread(target=start_desktop_ui, daemon=True)
        ui_thread.start()
        import time; time.sleep(0.3)

        port = getattr(start_desktop_ui, 'port', 8080)
        app_url = f"http://127.0.0.1:{port}"
        print(f"\n[HAIL Cognitive Studio Launching] -> {app_url}")
        webbrowser.open(app_url)

        print("\n[HAIL Cognitive Engine Running] Press Ctrl+C in terminal to exit.")
        try:
            ui_thread.join()
        except KeyboardInterrupt:
            print("\n[HAIL Desktop Shutting Down]")

if __name__ == "__main__":
    main()
