"""
Unit tests for HydrusOpt Cognem Tokenizer.
"""

import unittest
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hcl_core.cognems import build_cognem_vocab, CognemTokenizer, get_phrase_phase


class MockTokenizer:
    """Mock base tokenizer for testing."""
    
    def __init__(self):
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2
        self.vocab = {
            'the': 10, 'user': 11, 'wants': 12, 'to': 13,
            'authenticate': 14, 'with': 15, 'oauth': 16,
            'error': 17, 'handling': 18, 'try': 19, 'catch': 20,
            'database': 21, 'connection': 22, 'pool': 23,
            'hello': 24, 'world': 25, ' ': 26,
        }
    
    def encode(self, text, add_special_tokens=True):
        words = text.lower().strip().split()
        ids = []
        for w in words:
            clean = w.strip('.,!?')
            if clean in self.vocab:
                ids.append(self.vocab[clean])
            else:
                # Char fallback
                for c in clean:
                    ids.append(ord(c) % 100 + 100)
        if add_special_tokens:
            return [self.bos_token_id] + ids + [self.eos_token_id]
        return ids
    
    def decode(self, token_ids, skip_special_tokens=True):
        reverse = {v: k for k, v in self.vocab.items()}
        parts = []
        for tid in token_ids:
            if skip_special_tokens and tid in [self.bos_token_id, self.eos_token_id, self.pad_token_id]:
                continue
            if tid in reverse:
                parts.append(reverse[tid])
            else:
                parts.append(chr(tid - 100))
        return " ".join(parts)


class TestBuildCognemVocab(unittest.TestCase):
    
    def test_basic_extraction(self):
        texts = [
            "the user wants to authenticate with oauth",
            "the user wants to authenticate with oauth",
            "error handling with try catch",
            "database connection pool",
            "database connection pool",
            "database connection pool",
        ]
        
        vocab = build_cognem_vocab(texts, min_freq=2, max_phrase_len=3)
        
        # Should extract frequent phrases
        self.assertIn("the user wants", vocab)
        self.assertIn("database connection pool", vocab)
        self.assertNotIn("connection pool", vocab)
        
        # IDs should start at 50000
        self.assertEqual(vocab["database connection pool"], 50000)
        
    def test_min_freq_filter(self):
        texts = [
            "rare phrase here",
            "common phrase here",
            "common phrase here",
            "common phrase here",
        ]
        
        vocab = build_cognem_vocab(texts, min_freq=3, max_phrase_len=2)
        
        self.assertNotIn("rare phrase", vocab)
        self.assertIn("common phrase", vocab)
    
    def test_subsumption_avoidance(self):
        """Longer phrases should subsume shorter ones."""
        texts = [
            "the user wants to authenticate",
            "the user wants to authenticate",
            "user wants to",
            "user wants to",
        ]
        
        vocab = build_cognem_vocab(texts, min_freq=2, max_phrase_len=5)
        
        # "the user wants to authenticate" is longer and frequent
        self.assertIn("the user wants to authenticate", vocab)
        # "user wants to" should NOT be included (subsumed by longer phrase)
        self.assertNotIn("user wants to", vocab)
    
    def test_stopwords(self):
        texts = [
            "the user wants to authenticate",
            "the user wants to authenticate",
        ]
        
        vocab = build_cognem_vocab(
            texts, min_freq=1, max_phrase_len=3,
            stopwords={"the", "to", "with"}
        )
        
        # Phrases dominated by stopwords should be excluded
        self.assertNotIn("the user wants", vocab)
        self.assertIn("user wants to", vocab)  # Only 1 stopword out of 3
    
    def test_empty_corpus(self):
        vocab = build_cognem_vocab([], min_freq=1)
        self.assertEqual(vocab, {})


class TestCognemTokenizer(unittest.TestCase):
    
    def setUp(self):
        self.base = MockTokenizer()
        self.vocab = {
            "the user": 50000,
            "user wants": 50001,
            "authenticate with oauth": 50002,
            "error handling": 50003,
            "database connection pool": 50004,
        }
        self.tokenizer = CognemTokenizer(self.base, self.vocab)
    
    def test_encode_cognem_match(self):
        text = "the user wants to authenticate with oauth"
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Should match "the user" as first cognem
        self.assertEqual(tokens[0], 50000)
        
        # Should match "authenticate with oauth" later
        self.assertIn(50002, tokens)
    
    def test_encode_fallback_to_base(self):
        text = "hello world"
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # No cognems match, should fall back to base
        self.assertIn(24, tokens)  # "hello"
        self.assertIn(25, tokens)  # "world"
    
    def test_encode_mixed(self):
        text = "the user wants to hello world"
        tokens, source = self.tokenizer.encode(text, add_special_tokens=False, return_base_tokens=True)
        
        # "the user" should be cognem
        self.assertEqual(tokens[0], 50000)
        
        # "hello world" should be base tokens
        base_ids = [tid for tid, _ in source if tid < 50000]
        self.assertIn(24, base_ids)
        self.assertIn(25, base_ids)
    
    def test_decode_cognem(self):
        tokens = [50000, 50002]  # "the user" + "authenticate with oauth"
        text = self.tokenizer.decode(tokens)
        
        self.assertIn("the user", text)
        self.assertIn("authenticate with oauth", text)
    
    def test_decode_base_tokens(self):
        tokens = [10, 11]  # "the" + "user" (base)
        text = self.tokenizer.decode(tokens)
        
        self.assertIn("the", text)
        self.assertIn("user", text)
    
    def test_decode_mixed(self):
        tokens = [50000, 10, 11]  # cognem + base
        text = self.tokenizer.decode(tokens)
        
        self.assertIn("the user", text)
        self.assertIn("the", text)
        self.assertIn("user", text)
    
    def test_special_tokens_handling(self):
        text = "hello world"
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        self.assertEqual(tokens[0], self.base.bos_token_id)
        self.assertEqual(tokens[-1], self.base.eos_token_id)
        
        decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
        self.assertNotIn(str(self.base.bos_token_id), decoded)
    
    def test_compression_ratio(self):
        text = "the user wants to authenticate with oauth"
        
        base_count = len(self.base.encode(text))
        cognem_count = len(self.tokenizer.encode(text, add_special_tokens=False))
        ratio = self.tokenizer.compression_ratio(text)
        
        self.assertGreater(ratio, 1.0)
        self.assertAlmostEqual(ratio, base_count / cognem_count, places=2)
    
    def test_add_cognem_dynamic(self):
        self.tokenizer.add_cognem("dynamic phrase", 60000)
        
        self.assertIn("dynamic phrase", self.tokenizer.vocab)
        self.assertEqual(self.tokenizer.vocab["dynamic phrase"], 60000)
        
        # Should be usable immediately
        tokens = self.tokenizer.encode("dynamic phrase", add_special_tokens=False)
        self.assertIn(60000, tokens)
    
    def test_save_and_load_vocab(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
        
        try:
            self.tokenizer.save_vocab(path)
            
            loaded = CognemTokenizer(self.base, vocab_path=path)
            self.assertEqual(loaded.vocab, self.tokenizer.vocab)
        finally:
            os.unlink(path)
    
    def test_token_breakdown(self):
        text = "the user wants to authenticate with oauth"
        breakdown = self.tokenizer.token_breakdown(text)
        
        self.assertIn('tokens', breakdown)
        self.assertIn('source_map', breakdown)
        self.assertIn('compression_ratio', breakdown)
        self.assertGreater(breakdown['cognem_count'], 0)
    
    def test_greedy_longest_match(self):
        """Verify longest phrase wins in greedy matching."""
        vocab = {
            "the": 50000,
            "the user": 50001,
            "the user wants": 50002,
        }
        tokenizer = CognemTokenizer(self.base, vocab)
        
        text = "the user wants to"
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Should match longest phrase "the user wants", not shorter ones
        self.assertEqual(tokens[0], 50002)
    
    def test_spacing_preservation(self):
        text = "the user wants to authenticate with oauth"
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        decoded = self.tokenizer.decode(tokens)
        
        # Should be readable, not glued together
        self.assertIn(" ", decoded)
        # Should not have double spaces from cognem boundaries
        self.assertNotIn("  ", decoded)
        
    def test_quantum_mode_interference(self):
        """Verify that quantum mode dynamically selects segmentations based on phase alignment."""
        # Overlapping vocab:
        # 1. "user authentication"
        # 2. "authentication flow"
        vocab = {
            "user authentication": 50000,
            "authentication flow": 50001,
            "user": 50002,
            "authentication": 50003,
            "flow": 50004
        }
        
        tokenizer = CognemTokenizer(self.base, vocab, mode="quantum", interference_strength=0.95)
        text = "user authentication flow"
        
        # Retrieve computed phases for both overlapping phrases
        phase_1 = get_phrase_phase("user authentication")
        phase_2 = get_phrase_phase("authentication flow")
        
        # When context aligns with phase_1, it should favor "user authentication" + "flow"
        tokens_a = tokenizer.encode(text, add_special_tokens=False, context_phase=phase_1)
        
        # When context aligns with phase_2, it should favor "user" + "authentication flow"
        tokens_b = tokenizer.encode(text, add_special_tokens=False, context_phase=phase_2)
        
        # Ensure that changing the context phase shifts the resolved token path
        self.assertNotEqual(tokens_a, tokens_b)
        
        # Verify specific tokens in both configurations
        self.assertIn(50000, tokens_a)  # Matches "user authentication"
        self.assertIn(50001, tokens_b)  # Matches "authentication flow"


class TestCognemIntegration(unittest.TestCase):
    """Integration tests with realistic patterns."""
    
    def test_code_documentation_patterns(self):
        texts = [
            "def user_authentication_flow():",
            "user authentication flow requires oauth2 token",
            "handle error with try catch block",
            "database connection pool exhausted",
            "the user wants to authenticate with oauth",
        ] * 5  # Repeat for frequency
        
        vocab = build_cognem_vocab(texts, min_freq=3, max_phrase_len=4)
        
        # Code patterns should emerge
        self.assertTrue(
            any("user" in p for p in vocab),
            "User-related phrases should be extracted"
        )
    
    def test_hydrusopt_log_patterns(self):
        """Simulate HydrusOpt conversation logs."""
        logs = [
            "GBPS grounding score is 0.85 for user query",
            "HIL metacognition layer activated reasoning chain",
            "safety guardrail triggered on harmful intent",
            "GBPS decay signal indicates stale context",
            "HIL reasoning chain completed in 3 steps",
            "user authentication via OAuth2 successful",
            "database connection pool size exceeded limit",
        ] * 10
        
        vocab = build_cognem_vocab(logs, min_freq=5, max_phrase_len=5)
        
        tokenizer = CognemTokenizer(MockTokenizer(), vocab)
        
        test_query = "GBPS grounding score is 0.85 for user query"
        ratio = tokenizer.compression_ratio(test_query)
        
        self.assertGreater(ratio, 1.0, "Should achieve compression on HydrusOpt logs")


if __name__ == '__main__':
    unittest.main()
