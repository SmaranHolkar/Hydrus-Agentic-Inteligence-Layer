"""
HydrusOpt Cognem Tokenizer
Top-down semantic compression via macro-tokenization.
Common phrases become single IDs, reducing context window usage.
Uses quantum-inspired phase-interference context matching and prefix trie lookup for O(L) time complexity.
"""

import re
import json
import math
import hashlib
from collections import Counter
from typing import List, Dict, Optional, Union, Tuple

def get_phrase_phase(phrase: str) -> float:
    """
    Generate a deterministic phase in [-pi, pi] for a phrase.
    Used for quantum-inspired phase interference matching.
    """
    h = hashlib.md5(phrase.encode('utf-8')).hexdigest()
    val = int(h[:8], 16)  # Use first 8 characters
    return (val / 0xFFFFFFFF) * 2 * 3.141592653589793 - 3.141592653589793


class TrieNode:
    def __init__(self):
        self.children = {}
        self.cognem_id = None
        self.phrase = None


class PhraseTrie:
    """Prefix Tree for O(L) phrase matching complexity."""
    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, phrase: str, cognem_id: int):
        node = self.root
        for char in phrase:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.cognem_id = cognem_id
        node.phrase = phrase
        
    def search_all_prefixes(self, text: str, start: int) -> List[Tuple[str, int, int]]:
        """
        Finds all prefixes matching text starting at position 'start'.
        Returns list of tuples (phrase, cognem_id, match_len).
        """
        results = []
        node = self.root
        for i in range(start, len(text)):
            char = text[i]
            if char not in node.children:
                break
            node = node.children[char]
            if node.cognem_id is not None:
                results.append((node.phrase, node.cognem_id, i - start + 1))
        return results


def build_cognem_vocab(
    texts: List[str],
    min_freq: int = 5,
    max_phrase_len: int = 5,
    vocab_start_id: int = 50000,
    stopwords: Optional[set] = None,
) -> Dict[str, int]:
    """
    Extract recurrent n-grams from corpus and build cognem vocabulary.
    
    Args:
        texts: List of strings (logs, chats, docs, etc.)
        min_freq: Minimum occurrences to promote to cognem
        max_phrase_len: Max words in a phrase
        vocab_start_id: First ID for cognems (above base vocab)
        stopwords: Words to ignore in phrases
    
    Returns:
        Dict mapping phrase -> cognem ID
    """
    if stopwords is None:
        stopwords = set()
    
    phrases = Counter()
    
    for text in texts:
        # Normalize: lowercase, strip extra whitespace
        text = " ".join(text.lower().split())
        words = text.split()
        
        if len(words) < 2:
            continue
            
        for n in range(2, max_phrase_len + 1):
            for i in range(len(words) - n + 1):
                phrase_words = words[i:i+n]
                
                # Skip phrases starting with a stopword or dominated by stopwords
                if phrase_words[0] in stopwords or sum(1 for w in phrase_words if w in stopwords) > n // 2:
                    continue
                    
                phrase = " ".join(phrase_words)
                phrases[phrase] += 1
    
    # Build vocab with subsumption check
    # Longer phrases get priority; don't include subphrases of already-included
    cognems = {}
    filtered_phrases = [p for p, c in phrases.most_common() if c >= min_freq]
    # Sort by length descending to ensure longer phrases are processed first for subsumption
    sorted_phrases = sorted(filtered_phrases, key=len, reverse=True)
    
    for phrase in sorted_phrases:
        # Check if this phrase is already subsumed by a longer cognem
        is_subsumed = False
        for existing in cognems:
            if phrase in existing and phrase != existing:
                is_subsumed = True
                break
        
        if not is_subsumed:
            cognems[phrase] = vocab_start_id + len(cognems)
    
    return cognems


class CognemTokenizer:
    """
    Wraps any base tokenizer with cognem (macro-token) compression.
    
    Encoding: greedy longest-match phrase substitution, fallback to base.
    Decoding: cognem expansion + base token reconstruction.
    """
    
    def __init__(
        self,
        base_tokenizer,
        cognem_vocab: Optional[Dict[str, int]] = None,
        vocab_path: Optional[str] = None,
        mode: str = "greedy",
        interference_strength: float = 0.5,
    ):
        """
        Args:
            base_tokenizer: Any tokenizer with .encode() and .decode()
            cognem_vocab: Dict phrase -> ID, or None if loading from path
            vocab_path: Path to saved cognem vocab JSON
            mode: Matching strategy ("greedy" or "quantum")
            interference_strength: Strength of phase-interference (0.0 to 1.0)
        """
        self.base = base_tokenizer
        self.mode = mode
        self.interference_strength = interference_strength
        
        if vocab_path:
            with open(vocab_path, 'r') as f:
                self.vocab = json.load(f)
        elif cognem_vocab:
            self.vocab = cognem_vocab
        else:
            self.vocab = {}
        
        self._update_phrase_index()
    
    def _update_phrase_index(self):
        """Rebuild sorted phrase list and prefix Trie after vocab changes."""
        self.phrases = sorted(self.vocab.keys(), key=len, reverse=True)
        self.reverse = {v: k for k, v in self.vocab.items()}
        
        # Build phrase prefix trie for O(L) search speed
        self.trie = PhraseTrie()
        for phrase, cid in self.vocab.items():
            self.trie.insert(phrase, cid)
    
    def save_vocab(self, path: str):
        """Save cognem vocabulary to JSON."""
        with open(path, 'w') as f:
            json.dump(self.vocab, f, indent=2)
    
    def add_cognem(self, phrase: str, cognem_id: Optional[int] = None):
        """Add a new cognem dynamically."""
        phrase = phrase.lower().strip()
        if phrase in self.vocab:
            return self.vocab[phrase]
        
        if cognem_id is None:
            cognem_id = max(self.vocab.values(), default=49999) + 1
        
        self.vocab[phrase] = cognem_id
        self._update_phrase_index()
        return cognem_id
    
    def _find_cognem_match(self, text: str, start: int) -> Tuple[Optional[int], int]:
        """
        Find longest cognem match starting at position.
        Returns (cognem_id, match_length) or (None, 0).
        """
        matches = self.trie.search_all_prefixes(text, start)
        if not matches:
            return None, 0
        # Sort by match length descending to take longest match
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[0][1], matches[0][2]
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_base_tokens: bool = False,
        context_phase: float = 0.0,
    ) -> Union[List[int], Tuple[List[int], List[Tuple[int, str]]]]:
        """
        Encode text with cognem compression.
        
        Args:
            text: Input string
            add_special_tokens: Passed to base tokenizer
            return_base_tokens: If True, also return base token mapping for debugging
            context_phase: Phase of the active grounding context for quantum mode
        
        Returns:
            List of token IDs, or (token_ids, [(id, source)]) if return_base_tokens
        """
        text_lower = text.lower()
        tokens = []
        source_map = []  # For debugging: (token_id, source)
        i = 0
        
        if self.mode == "greedy":
            while i < len(text_lower):
                # Try cognem match first
                cognem_id, match_len = self._find_cognem_match(text_lower, i)
                
                if cognem_id is not None:
                    tokens.append(cognem_id)
                    source_map.append((cognem_id, text_lower[i:i+match_len]))
                    i += match_len
                    continue
                
                # Fallback: find contiguous chunk to send to base tokenizer
                chunk_start = i
                chunk_end = min(i + 50, len(text_lower))
                
                # Look ahead to find next cognem boundary
                for j in range(i + 1, min(i + 50, len(text_lower))):
                    _, ml = self._find_cognem_match(text_lower, j)
                    if ml > 0:
                        chunk_end = j
                        break
                    # Also break on sentence boundaries
                    if text_lower[j-1] in '.!?\n':
                        chunk_end = j
                        break
                
                chunk = text[chunk_start:chunk_end]  # Use original casing for base
                
                # Encode chunk with base tokenizer
                base_ids = self.base.encode(chunk, add_special_tokens=False)
                
                # Filter out duplicate special tokens if not at start
                if tokens and not add_special_tokens:
                    # Remove BOS if present and not first
                    if hasattr(self.base, 'bos_token_id') and base_ids and base_ids[0] == self.base.bos_token_id:
                        base_ids = base_ids[1:]
                
                tokens.extend(base_ids)
                source_map.extend([(bid, chunk) for bid in base_ids])
                i = chunk_end
        
        elif self.mode == "quantum":
            # Quantum-inspired path selection over overlapping candidate superposition
            L = len(text_lower)
            dp = [-1e9] * (L + 1)
            dp[0] = 0.0
            parent = [None] * (L + 1)
            
            for curr_i in range(L):
                if dp[curr_i] < -1e8:
                    continue
                
                # Find all prefix matches at the current position
                prefixes = self.trie.search_all_prefixes(text_lower, curr_i)
                for phrase, cognem_id, match_len in prefixes:
                    phase = get_phrase_phase(phrase)
                    # Constructive/destructive wave interference with context phase
                    interference = 1.0 + self.interference_strength * math.cos(phase - context_phase)
                    w = (match_len * 10.0) * interference
                    
                    if dp[curr_i] + w > dp[curr_i + match_len]:
                        dp[curr_i + match_len] = dp[curr_i] + w
                        parent[curr_i + match_len] = (curr_i, cognem_id, phrase)
                
                # Fallback step
                w_fallback = 1.0
                if dp[curr_i] + w_fallback > dp[curr_i + 1]:
                    dp[curr_i + 1] = dp[curr_i] + w_fallback
                    parent[curr_i + 1] = (curr_i, None, text_lower[curr_i])
            
            # Reconstruct optimal path
            path = []
            curr = L
            while curr > 0:
                prev_i, token_id, source = parent[curr]
                path.append((prev_i, curr, token_id, source))
                curr = prev_i
            path.reverse()
            
            # Process resolved path chunks
            path_idx = 0
            while path_idx < len(path):
                start_idx, end_idx, token_id, source = path[path_idx]
                if token_id is not None:
                    tokens.append(token_id)
                    source_map.append((token_id, text[start_idx:end_idx]))
                    path_idx += 1
                else:
                    # Group contiguous fallback characters
                    chunk_start = start_idx
                    chunk_end = end_idx
                    next_idx = path_idx + 1
                    while next_idx < len(path) and path[next_idx][2] is None:
                        chunk_end = path[next_idx][1]
                        next_idx += 1
                    
                    chunk = text[chunk_start:chunk_end]
                    base_ids = self.base.encode(chunk, add_special_tokens=False)
                    
                    if tokens and not add_special_tokens:
                        if hasattr(self.base, 'bos_token_id') and base_ids and base_ids[0] == self.base.bos_token_id:
                            base_ids = base_ids[1:]
                            
                    tokens.extend(base_ids)
                    source_map.extend([(bid, chunk) for bid in base_ids])
                    path_idx = next_idx

        # Add special tokens if requested and base tokenizer has them
        if add_special_tokens:
            if hasattr(self.base, 'bos_token_id') and self.base.bos_token_id is not None:
                if not tokens or tokens[0] != self.base.bos_token_id:
                    tokens.insert(0, self.base.bos_token_id)
            if hasattr(self.base, 'eos_token_id') and self.base.eos_token_id is not None:
                if not tokens or tokens[-1] != self.base.eos_token_id:
                    tokens.append(self.base.eos_token_id)
        
        if return_base_tokens:
            return tokens, source_map
        return tokens
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back to text.
        
        Reconstructs spacing by tracking whether previous output ended with space.
        """
        parts = []
        prev_ended_space = False
        
        for tid in token_ids:
            # Skip special tokens
            if skip_special_tokens:
                if hasattr(self.base, 'bos_token_id') and tid == self.base.bos_token_id:
                    continue
                if hasattr(self.base, 'eos_token_id') and tid == self.base.eos_token_id:
                    continue
                if hasattr(self.base, 'pad_token_id') and tid == self.base.pad_token_id:
                    continue
            
            if tid in self.reverse:
                # Cognem expansion
                text = self.reverse[tid]
                # Add space if needed and not already present
                if parts and not prev_ended_space and not text.startswith(' '):
                    parts.append(' ')
                parts.append(text)
                prev_ended_space = text.endswith(' ')
            else:
                # Base token
                try:
                    text = self.base.decode([tid])
                except:
                    text = self.base.decode([tid], skip_special_tokens=skip_special_tokens)
                
                # Handle BPE spacing: base decoders often return 'Ġword' or ' word'
                # Normalize spacing
                if text.startswith('Ġ') or text.startswith(' '):
                    if parts and not prev_ended_space:
                        parts.append(' ')
                    text = text.lstrip('Ġ').lstrip(' ')
                
                parts.append(text)
                prev_ended_space = text.endswith(' ') or text.endswith('\n')
        
        return "".join(parts)
    
    def compression_ratio(self, text: str) -> float:
        """
        Calculate compression ratio: base_tokens / cognem_tokens.
        Higher = better compression.
        """
        base_tokens = len(self.base.encode(text))
        cognem_tokens = len(self.encode(text, add_special_tokens=False))
        
        if cognem_tokens == 0:
            return 0.0
        
        return base_tokens / cognem_tokens
    
    def token_breakdown(self, text: str) -> Dict:
        """
        Debug helper: show how text was tokenized.
        """
        tokens, source_map = self.encode(text, return_base_tokens=True)
        
        cognem_count = sum(1 for tid, _ in source_map if tid in self.reverse)
        base_count = len(source_map) - cognem_count
        
        return {
            'text': text,
            'tokens': tokens,
            'source_map': source_map,
            'cognem_count': cognem_count,
            'base_count': base_count,
            'compression_ratio': self.compression_ratio(text),
        }
