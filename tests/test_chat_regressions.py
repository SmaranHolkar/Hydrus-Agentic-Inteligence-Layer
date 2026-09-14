import importlib.util
import unittest
from pathlib import Path


def _load_desktop_app_module():
    module_path = Path(__file__).resolve().parents[1] / "HAIL" / "distros" / "hail-desktop" / "app.py"
    spec = importlib.util.spec_from_file_location("hail_desktop_app", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestDesktopChatRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _load_desktop_app_module()

    def test_email_lost_package_generates_draft(self):
        reply = self.app._generate_local_chat_fallback(
            intent="CHAT",
            prompt="help me write an email about my lost package",
            memories=[],
            chat_history=[]
        )
        self.assertIn("Missing Package Inquiry", reply)
        self.assertIn("Order #[Your Order Number]", reply)

    def test_route_intent_for_email_is_chat(self):
        intent = self.app._route_intent("Can you write an email to customer support for me?")
        self.assertEqual(intent, "CHAT")

    def test_screenplay_text_not_misclassified_as_character_brief(self):
        screenplay_prompt = (
            "INT. CONTROL ROOM - NIGHT\n"
            "FADE IN:\n"
            "NARRATOR (V.O.) The walls are watching.\n"
            "CUT TO: Sector 2 where the broadcast repeats.\n"
            "SCENE: A woman repeats the same movement in a loop."
        )
        reply = self.app._generate_local_chat_fallback(
            intent="CHAT",
            prompt=screenplay_prompt,
            memories=["project: ensers", "writing a tv show"],
            chat_history=[]
        )
        self.assertIn("episode excerpt", reply.lower())

    def test_episode_followup_keeps_context(self):
        previous_screenplay = (
            "INT. SECTOR GATE - NIGHT\n"
            "FADE IN\n"
            "NARRATOR (V.O.) This tape should not exist.\n"
            "SCENE: The pulse on the wall starts matching heartbeats."
        )
        history = [
            {"role": "user", "text": previous_screenplay},
            {"role": "assistant", "text": "This reads like an episode excerpt."},
        ]
        reply = self.app._generate_local_chat_fallback(
            intent="CHAT",
            prompt="That is episode 3",
            memories=["writing a series"],
            chat_history=history
        )
        self.assertIn("Episode 3", reply)

    def test_character_followup_uses_named_lead(self):
        history = [
            {"role": "assistant", "text": "Let us brainstorm your tv show character arc."},
        ]
        reply = self.app._generate_local_chat_fallback(
            intent="CHAT",
            prompt="His name is Elias",
            memories=["writing a tv show"],
            chat_history=history
        )
        self.assertIn("Elias", reply)

    def test_route_intent_short_followup_uses_recent_context(self):
        history = [
            {"role": "assistant", "text": "This scene works well for your episode draft."},
            {"role": "user", "text": "that is episode 3"},
        ]
        intent = self.app._route_intent("and continue", chat_history=history)
        self.assertEqual(intent, "CHAT")

    def test_longform_token_budget_is_higher(self):
        prompt = "Write a detailed long-form episode outline with character arcs, 8 beats, and a full thematic progression."
        tokens = self.app._recommended_max_new_tokens(prompt, "CHAT")
        self.assertGreaterEqual(tokens, 500)

    def test_runtime_model_claim_is_honest_for_synthetic_mode(self):
        claim = self.app._runtime_model_claim(
            "moe:qwen3-35b-a3b",
            "qwen3-35b-a3b",
            {
                "execution": {
                    "streaming_generation_verified": False,
                    "weights_mode": "synthetic_fixture"
                },
                "prefetcher": {"metrics": {"query_hit_rate": 0.0}}
            }
        )
        self.assertIn("not verified full 35B streamed language generation", claim)

    def test_runtime_model_claim_for_verified_local_blobs(self):
        claim = self.app._runtime_model_claim(
            "moe:qwen3-35b-a3b",
            "qwen3-35b-a3b",
            {
                "execution": {
                    "streaming_generation_verified": True,
                    "weights_mode": "real_local_blobs"
                },
                "prefetcher": {"metrics": {"query_hit_rate": 0.73}}
            }
        )
        self.assertIn("verified local expert blobs", claim)
        self.assertIn("0.73", claim)

    def test_factual_confidence_estimator_marks_uncertain_text_low(self):
        conf = self.app._estimate_factual_confidence(
            "I am not sure, maybe this is correct but I could not find a direct source."
        )
        self.assertEqual(conf["label"], "low")

    def test_factual_confidence_estimator_marks_clear_text_high(self):
        conf = self.app._estimate_factual_confidence(
            "The Airbus A350-900 typically achieves very strong fuel efficiency on long-haul routes due to modern composite structure and efficient Rolls-Royce Trent XWB engines."
        )
        self.assertIn(conf["label"], ["high", "medium"])


if __name__ == "__main__":
    unittest.main()
