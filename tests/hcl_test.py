import sys
import unittest
import numpy as np
import os
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hcl_core.hcl import (
    NodeType, Node, MemoryGraph, KNNIndex,
    GroundedBeliefPathSearch, BeliefTransitionModel, WarmCache,
    ModeController, CryptoHelper, CognitiveProfile,
    HydrusOptEntropyScorer, HCL, resolve_state,
    DreamState, AXIOM_PROPERTIES, generate_with_hcl,
    is_personal_query, should_fact_check, is_social_utterance, is_question
)

class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 2
    def decode(self, token_ids, skip_special_tokens=True):
        return "mock decoded text tail"
    def encode(self, text, return_tensors=None):
        return [1, 2, 3]
    def __call__(self, text, return_tensors=None, **kwargs):
        import torch
        return {"input_ids": torch.tensor([[1, 2, 3]])}

class MockModel:
    def __init__(self):
        import torch
        self.device = torch.device("cpu")
    def __call__(self, input_ids, **kwargs):
        class MockOutputs:
            def __init__(self):
                import torch
                seq_len = input_ids.shape[1]
                self.logits = torch.zeros((1, seq_len, 50))
        return MockOutputs()

class TestHCLCore(unittest.TestCase):
    def setUp(self):
        for f in os.listdir("."):
            if (f.startswith("lattice_") or f.startswith("profile_")) and f.endswith(".hcl"):
                try:
                    os.remove(f)
                except Exception:
                    pass

    def tearDown(self):
        for f in os.listdir("."):
            if (f.startswith("lattice_") or f.startswith("profile_")) and f.endswith(".hcl"):
                try:
                    os.remove(f)
                except Exception:
                    pass

    def test_node_creation(self):
        emb = np.random.rand(768)
        node = Node(NodeType.EPISODIC, emb, "episodic test node")
        self.assertEqual(node.node_type, NodeType.EPISODIC)
        self.assertEqual(node.raw_summary, "episodic test node")
        self.assertEqual(node.weight, 0.8)
        self.assertEqual(node.state, 1)

    def test_crypto_helper(self):
        plaintext = "hcl_secret_user_profile_payload_2026"
        password = "SuperSecretPassphrase123"
        
        # Test valid encrypt/decrypt
        encrypted = CryptoHelper.encrypt(plaintext, password)
        decrypted = CryptoHelper.decrypt(encrypted, password)
        self.assertEqual(plaintext, decrypted)
        
        # Test wrong password fails
        with self.assertRaises(ValueError):
            CryptoHelper.decrypt(encrypted, "WrongPassphrase")
            
        # Test corrupted data fails
        corrupted = bytearray(encrypted)
        corrupted[20] ^= 0xFF  # corrupt a byte in the MAC or ciphertext
        with self.assertRaises(ValueError):
            CryptoHelper.decrypt(bytes(corrupted), password)

    def test_cognitive_profile_load_save(self):
        profile = CognitiveProfile("user_test_99")
        profile.entropy_baseline = 0.45
        profile.expertise_levels["science"] = 0.85
        profile.recurring_clusters = [1, 2, 5]
        
        filepath = "test_profile.hcl"
        passphrase = "HclPassphrase2026"
        
        try:
            profile.save(filepath, passphrase)
            self.assertTrue(os.path.exists(filepath))
            
            loaded = CognitiveProfile.load(filepath, passphrase)
            self.assertEqual(loaded.user_id, "user_test_99")
            self.assertEqual(loaded.entropy_baseline, 0.45)
            self.assertEqual(loaded.expertise_levels["science"], 0.85)
            self.assertEqual(loaded.recurring_clusters, [1, 2, 5])
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_knn_index_flat_and_lsh(self):
        index = KNNIndex(dim=768, lsh_bits=4)
        
        # 1. Flat index test (<= 100 nodes)
        nodes = []
        for i in range(10):
            emb = np.random.rand(768)
            emb /= np.linalg.norm(emb)
            node = Node(NodeType.EPISODIC, emb, f"node_{i}", weight=0.5)
            nodes.append(node)
            index.update(node)
            
        query_emb = np.random.rand(768)
        query_emb /= np.linalg.norm(query_emb)
        
        results = index.query(query_emb, k=3)
        self.assertEqual(len(results), 3)
        
        # 2. Scaling LSH index test (> 100 nodes)
        for i in range(150):
            emb = np.random.rand(768)
            emb /= np.linalg.norm(emb)
            node = Node(NodeType.SEMANTIC, emb, f"lsh_node_{i}", weight=0.6)
            index.update(node)
            
        results_lsh = index.query(query_emb, k=5)
        self.assertEqual(len(results_lsh), 5)

    def test_memory_graph_clustering(self):
        graph = MemoryGraph()
        hcl_obj = HCL(MockModel(), MockTokenizer(), mode="balanced")
        hcl_obj.HMG = graph
        
        # Add nodes and assert cluster assignments
        emb1 = np.ones(768)
        emb1 /= np.linalg.norm(emb1)
        node1 = Node(NodeType.EPISODIC, emb1, "cluster node 1")
        hcl_obj.assign_to_cluster(node1)
        graph.insert(node1)
        
        self.assertIsNotNone(node1.cluster_id)
        self.assertIn(node1.id, graph.clusters[node1.cluster_id])
        
        # Node 2 is highly similar, should merge into same cluster
        node2 = Node(NodeType.EPISODIC, emb1, "cluster node 2")
        hcl_obj.assign_to_cluster(node2)
        graph.insert(node2)
        self.assertEqual(node1.cluster_id, node2.cluster_id)
        
        # Node 3 is orthogonal/different, should form new cluster
        emb3 = np.ones(768)
        emb3[384:] = -1.0
        emb3 /= np.linalg.norm(emb3)
        node3 = Node(NodeType.EPISODIC, emb3, "different cluster node")
        hcl_obj.assign_to_cluster(node3)
        graph.insert(node3)
        self.assertNotEqual(node1.cluster_id, node3.cluster_id)

    def test_three_state_resolution(self):
        node = Node(NodeType.BELIEF, np.random.rand(768), "state test")
        
        # State 1: weight > 0.65
        node.weight = 0.85
        resolve_state(node, queried=False, steps_since_query=0)
        self.assertEqual(node.state, 1)
        
        # State 0: weight <= 0.20
        node.weight = 0.15
        resolve_state(node, queried=False, steps_since_query=0)
        self.assertEqual(node.state, 0)
        
        # State 2: 0.20 < weight <= 0.65
        node.weight = 0.50
        resolve_state(node, queried=False, steps_since_query=1)
        self.assertEqual(node.state, 2)
        
        # Collapse to State 0 if steps_since_query > N_collapse (e.g. 5)
        resolve_state(node, queried=False, steps_since_query=10)
        self.assertEqual(node.state, 0)
        
        # Resolve to State 1 on query
        node.weight = 0.50
        node.state = 2
        resolve_state(node, queried=True, steps_since_query=1)
        self.assertEqual(node.state, 1)
        self.assertEqual(node.weight, 0.65) # 0.50 + 0.15 (Î´_boost)

    def test_radix_sort_retrieval(self):
        """Nodes written via hcl.write() are retrievable through the full SML pipeline."""
        hcl_obj = HCL(MockModel(), MockTokenizer(), mode="balanced")
        for i in range(1, 6):
            emb = np.ones(768)
            emb /= np.linalg.norm(emb)
            node = Node(NodeType.BELIEF, emb, f"node_{i}",
                        weight=round(0.2 * i, 1), confidence=0.9, state=1)
            hcl_obj.write(node)
        query = np.ones(768)
        query /= np.linalg.norm(query)
        retrieved = hcl_obj.retrieve(query, k=5)
        self.assertGreater(len(retrieved), 0,
                           "Expected at least one node to be retrieved via SML")

    def test_hcl_autosave(self):
        import torch
        os.environ["HCL_PROFILE_KEY"] = "TestKey123"
        user_id = "test_autosave_user"
        profile_path = f"profile_{user_id}.hcl"
        if os.path.exists(profile_path):
            os.remove(profile_path)
            
        try:
            # Init HCL in persistent mode
            hcl_obj = HCL(MockModel(), MockTokenizer(), mode="persistent", user_id=user_id)
            self.assertEqual(hcl_obj.profile.session_count, 1)
            self.assertTrue(os.path.exists(profile_path))
            
            # Step once to verify inference step increment
            context = torch.zeros((1, 10), dtype=torch.long)
            token_dist = torch.zeros((1, 50))
            hcl_obj.on_generation_step(context, token_dist)
            
            self.assertEqual(hcl_obj.profile.total_inference_steps, 1)
            
            # Load from file to verify decryption and data consistency
            loaded_profile = CognitiveProfile.load(profile_path, "TestKey123")
            self.assertEqual(loaded_profile.session_count, 1)
            self.assertEqual(loaded_profile.total_inference_steps, 1)
        finally:
            if os.path.exists(profile_path):
                os.remove(profile_path)
            os.environ.pop("HCL_PROFILE_KEY", None)

    def test_dynamic_uncertainty_threshold(self):
        hcl_obj = HCL(MockModel(), MockTokenizer(), mode="balanced")
        base = 0.75

        # Creative prompt should increase threshold
        t_creative = hcl_obj.get_dynamic_uncertainty_threshold("Write a poem about Mars", "Sure,", base)
        self.assertGreater(t_creative, base)

        # Factual constraint prompt should decrease threshold
        t_factual = hcl_obj.get_dynamic_uncertainty_threshold("What is the exact population of Paris?", "Paris is a very large city", base)
        self.assertLess(t_factual, base)

        # List context in assistant response should increase threshold
        t_list = hcl_obj.get_dynamic_uncertainty_threshold("Tell me about blue", "feelings of trust, reliability, and", base)
        self.assertGreater(t_list, base)

        # Assertion context should decrease threshold
        t_assert = hcl_obj.get_dynamic_uncertainty_threshold("Tell me about blue", "The color blue is blue", base)
        self.assertLess(t_assert, base)

    def test_user_facts_extraction(self):
        hcl_obj = HCL(MockModel(), MockTokenizer(), mode="balanced")
        
        # Test direct extraction method
        extracted = hcl_obj.user_facts.extract_from_turn("My favorite color is blue")
        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0][0], "favourite_color")
        self.assertEqual(extracted[0][1], "blue")
        
        extracted_struggle = hcl_obj.user_facts.extract_from_turn("I struggle with algebra")
        self.assertEqual(len(extracted_struggle), 1)
        self.assertEqual(extracted_struggle[0][0], "struggle_topic")
        self.assertEqual(extracted_struggle[0][1], "algebra")
        
        # Test end-to-end via generate_with_hcl (mocked)
        hcl_obj.hcl_lightweight = True  # force save check / triggers save_profile if profile exists
        generate_with_hcl("My favorite color is green", MockModel(), MockTokenizer(), hcl_obj, max_new_tokens=2)
        
        self.assertIn("favourite_color", hcl_obj.user_facts.facts)
        self.assertEqual(hcl_obj.user_facts.facts["favourite_color"]["value"], "green")

    def test_user_facts_retrieval(self):
        hcl_obj = HCL(MockModel(), MockTokenizer(), mode="balanced")
        hcl_obj.user_facts.add_fact("favourite_color", "blue")
        hcl_obj.user_facts.add_fact("struggle_topic", "calculus")
        
        # Test exact match
        hcl_obj.current_query_text = "What is my favorite color?"
        context = hcl_obj.format_injection([])
        self.assertIn("- User fact: The user's favorite color is 'blue'.", context)
        self.assertNotIn("calculus", context)
        
        # Test struggle query
        hcl_obj.current_query_text = "I need help with calculus"
        context = hcl_obj.format_injection([])
        self.assertIn("- User fact: The user's struggle topic is 'calculus'.", context)
        self.assertNotIn("blue", context)

    def test_user_facts_persistence(self):
        os.environ["HCL_PROFILE_KEY"] = "TestPersistenceKey123"
        user_id = "test_persistence_user"
        profile_path = f"profile_{user_id}.hcl"
        if os.path.exists(profile_path):
            os.remove(profile_path)
            
        try:
            hcl_obj = HCL(MockModel(), MockTokenizer(), mode="persistent", user_id=user_id)
            hcl_obj.user_facts.add_fact("favourite_color", "yellow")
            hcl_obj.save_profile()
            
            # Load again in new HCL instance
            hcl_new = HCL(MockModel(), MockTokenizer(), mode="persistent", user_id=user_id)
            self.assertIn("favourite_color", hcl_new.user_facts.facts)
            self.assertEqual(hcl_new.user_facts.facts["favourite_color"]["value"], "yellow")
        finally:
            if os.path.exists(profile_path):
                os.remove(profile_path)
            os.environ.pop("HCL_PROFILE_KEY", None)

    def test_intent_aware_routing(self):
        # 1. Test is_personal_query
        self.assertTrue(is_personal_query("what is my favorite color"))
        self.assertTrue(is_personal_query("do you remember my name?"))
        self.assertFalse(is_personal_query("what is the mass of the Sun?"))
        self.assertFalse(is_personal_query("Tell me about the Antonov An-225 plane"))

        # 2. Test is_social_utterance
        self.assertTrue(is_social_utterance("My favourite plane is the a350-1000"))
        self.assertTrue(is_social_utterance("I've flown the A380 as well"))
        self.assertFalse(is_social_utterance("what is the capital of France"))

        # 3. Test is_question
        self.assertTrue(is_question("how many moons does jupiter have?"))
        self.assertTrue(is_question("What is my favorite color?"))
        self.assertTrue(is_question("is this working"))
        self.assertFalse(is_question("my favorite color is blue"))
        self.assertFalse(is_question("island is beautiful"))

        # 4. Test should_fact_check
        self.assertFalse(should_fact_check("Your favorite color is blue.", "what is my favorite color"))
        self.assertFalse(should_fact_check("You told me you love pizza.", "do you remember my favorite food"))
        self.assertFalse(should_fact_check("The A380 is also a popular choice for its luxurious design.", "My favourite plane is the a350-1000... I've flown the A380 as well"))
        self.assertTrue(should_fact_check("Paris is the capital of France.", "what is the capital of France"))


# =============================================================================
# Â§3.6  Dynamic Type Evolution
# =============================================================================

class TestDynamicTypeEvolution(unittest.TestCase):
    """Tests for Â§3.6: EPISODIC â†’ SEMANTIC â†’ BELIEF â†’ AXIOM promotion ladder."""

    def setUp(self):
        for f in os.listdir("."):
            if (f.startswith("lattice_") or f.startswith("profile_")) and f.endswith(".hcl"):
                try:
                    os.remove(f)
                except Exception:
                    pass
        self.hcl = HCL(MockModel(), MockTokenizer(), mode="balanced")

    def tearDown(self):
        for f in os.listdir("."):
            if (f.startswith("lattice_") or f.startswith("profile_")) and f.endswith(".hcl"):
                try:
                    os.remove(f)
                except Exception:
                    pass

    def _emb(self):
        e = np.random.rand(768)
        return e / np.linalg.norm(e)

    def test_episodic_to_semantic_promotion(self):
        node = Node(NodeType.EPISODIC, self._emb(), "episodic → semantic",
                    abstraction_score=0.8, access_count=11)
        self.assertTrue(self.hcl.promote_node_type(node))
        self.assertEqual(node.node_type, NodeType.SEMANTIC)

    def test_episodic_not_promoted_when_access_count_low(self):
        node = Node(NodeType.EPISODIC, self._emb(), "too few accesses",
                    abstraction_score=0.8, access_count=5)
        self.assertFalse(self.hcl.promote_node_type(node))
        self.assertEqual(node.node_type, NodeType.EPISODIC)

    def test_episodic_not_promoted_when_abstraction_score_low(self):
        node = Node(NodeType.EPISODIC, self._emb(), "low abstraction",
                    abstraction_score=0.5, access_count=20)
        self.assertFalse(self.hcl.promote_node_type(node))
        self.assertEqual(node.node_type, NodeType.EPISODIC)

    def test_semantic_to_belief_promotion(self):
        node = Node(NodeType.SEMANTIC, self._emb(), "semantic → belief",
                    confidence=0.95)
        self.assertTrue(self.hcl.promote_node_type(node))
        self.assertEqual(node.node_type, NodeType.BELIEF)

    def test_belief_to_axiom_promotion(self):
        node = Node(NodeType.BELIEF, self._emb(), "belief → axiom",
                    confidence=0.95, verification_count=101,
                    provenance=["src_A", "src_B", "src_C"])
        self.assertTrue(self.hcl.promote_node_type(node))
        self.assertEqual(node.node_type, NodeType.AXIOM)
        self.assertEqual(node.confidence, 1.0)

    def test_axiom_promotion_blocked_by_contradiction(self):
        node = Node(NodeType.BELIEF, self._emb(), "has contradictions",
                    confidence=0.95, verification_count=101,
                    provenance=["s1", "s2", "s3"],
                    contradictions=["some_node_id"])
        self.assertFalse(self.hcl.promote_node_type(node))
        self.assertEqual(node.node_type, NodeType.BELIEF)

    def test_axiom_promotion_blocked_by_low_source_diversity(self):
        node = Node(NodeType.BELIEF, self._emb(), "only 2 sources",
                    confidence=0.95, verification_count=101,
                    provenance=["src_A", "src_B"])  # < source_diversity_min=3
        self.assertFalse(self.hcl.promote_node_type(node))
        self.assertEqual(node.node_type, NodeType.BELIEF)

    def test_axiom_demotion_by_contradiction_count(self):
        node = Node(NodeType.AXIOM, self._emb(), "axiom demotion", confidence=1.0)
        node.contradictions = ["c1", "c2", "c3"]
        self.assertTrue(self.hcl.demote_axiom(node, reason='contradiction'))
        self.assertEqual(node.node_type, NodeType.BELIEF)
        self.assertLess(node.confidence, 1.0)
        self.assertIn('[DEMOTED FROM AXIOM]', node.raw_summary)

    def test_axiom_not_demoted_with_only_one_contradiction(self):
        """Demotion threshold is 3; with 1 contradiction it should not fire."""
        node = Node(NodeType.AXIOM, self._emb(), "still axiom", confidence=1.0)
        node.contradictions = ["c1"]
        self.assertFalse(self.hcl.demote_axiom(node, reason='contradiction'))
        self.assertEqual(node.node_type, NodeType.AXIOM)

    def test_axiom_demotion_by_user_explicit(self):
        node = Node(NodeType.AXIOM, self._emb(), "user challenged", confidence=1.0)
        node.contradictions = ["c1"]  # only 1, but user_explicit overrides
        self.assertTrue(self.hcl.demote_axiom(node, reason='user_explicit'))
        self.assertEqual(node.node_type, NodeType.BELIEF)

    def test_non_axiom_cannot_be_demoted(self):
        node = Node(NodeType.BELIEF, self._emb(), "plain belief", confidence=0.8)
        self.assertFalse(self.hcl.demote_axiom(node))
        self.assertEqual(node.node_type, NodeType.BELIEF)

    def test_axiom_properties_governance_constants(self):
        """Verify governance constants match the Â§3.6 spec."""
        self.assertEqual(AXIOM_PROPERTIES['promotion_threshold'], 100)
        self.assertEqual(AXIOM_PROPERTIES['demotion_threshold'],  3)
        self.assertEqual(AXIOM_PROPERTIES['source_diversity_min'], 3)
        self.assertEqual(AXIOM_PROPERTIES['max_axiom_lifetime'],  86400 * 30)


# =============================================================================
# Â§3.8  Dream-State Compression
# =============================================================================

class TestDreamState(unittest.TestCase):
    """Tests for Â§3.8: compress_to_dream and feel_deja_vu."""

    def _ds(self, threshold=0.3):
        return DreamState(dim=64, threshold=threshold)

    def _mems(self, n, base=None, noise=0.01):
        if base is None:
            base = np.ones(64, dtype=np.float32)
            base /= np.linalg.norm(base)
        return [{'embedding': base + np.random.randn(64).astype(np.float32) * noise,
                 'temperature': 0.4,
                 'summary': f'memory {i}'} for i in range(n)]

    def test_compress_empty_returns_minus_one(self):
        ds = self._ds()
        self.assertEqual(ds.compress_to_dream([]), -1)
        self.assertEqual(len(ds.dreams), 0)

    def test_compress_creates_dream_entry(self):
        ds = self._ds()
        dream_id = ds.compress_to_dream(self._mems(5))
        self.assertEqual(dream_id, 0)
        self.assertIn(0, ds.dreams)
        d = ds.dreams[0]
        for key in ('centroid', 'trigger_radius', 'abstract_summary', 'member_count'):
            self.assertIn(key, d)
        self.assertEqual(d['member_count'], 5)

    def test_sequential_compress_increments_id(self):
        ds = self._ds()
        id0 = ds.compress_to_dream(self._mems(2))
        id1 = ds.compress_to_dream(self._mems(2))
        self.assertEqual(id0, 0)
        self.assertEqual(id1, 1)
        self.assertEqual(len(ds.dreams), 2)

    def test_centroid_is_unit_normalised(self):
        ds = self._ds()
        ds.compress_to_dream(self._mems(4))
        centroid = ds.dreams[0]['centroid']
        self.assertAlmostEqual(float(np.linalg.norm(centroid)), 1.0, places=5)

    def test_abstract_summary_contains_source_snippets(self):
        ds = self._ds()
        mems = [{'embedding': np.random.randn(64).astype(np.float32),
                 'temperature': 0.3,
                 'summary': f'unique fact {i}'} for i in range(3)]
        ds.compress_to_dream(mems)
        summary = ds.dreams[0]['abstract_summary']
        self.assertTrue(len(summary) > 0)
        self.assertNotEqual(summary, 'impressionistic memory')

    def test_feel_deja_vu_hit_on_near_query(self):
        ds = self._ds()
        base = np.ones(64, dtype=np.float32)
        base /= np.linalg.norm(base)
        ds.compress_to_dream(self._mems(4, base=base, noise=0.01))
        result = ds.feel_deja_vu(base.copy())
        self.assertIsNotNone(result)
        self.assertEqual(result['type'], 'deja_vu')
        self.assertGreater(result['strength'], 0.0)
        self.assertIn('impression', result)
        self.assertIn('dream_id', result)

    def test_feel_deja_vu_miss_below_threshold(self):
        """A very distant query should not fire when threshold is high."""
        ds = self._ds(threshold=0.9)
        base = np.ones(64, dtype=np.float32)
        base /= np.linalg.norm(base)
        ds.compress_to_dream(self._mems(4, base=base, noise=0.01))
        result = ds.feel_deja_vu(-base.copy())
        self.assertIsNone(result)

    def test_multiple_dreams_returns_strongest_match(self):
        ds = self._ds()
        base = np.ones(64, dtype=np.float32)
        base /= np.linalg.norm(base)
        # Dream 0: antipodal (far from query)
        ds.compress_to_dream([{'embedding': -base.copy(),
                                'temperature': 0.3, 'summary': 'far memory'}])
        # Dream 1: close to query
        ds.compress_to_dream(self._mems(4, base=base, noise=0.01))
        result = ds.feel_deja_vu(base)
        if result is not None:
            self.assertEqual(result['dream_id'], 1,
                             "Should return the closest dream, not the furthest")


if __name__ == "__main__":
    unittest.main()
