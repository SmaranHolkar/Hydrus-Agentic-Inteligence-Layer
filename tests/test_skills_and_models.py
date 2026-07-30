"""
Tests for HAIL Core skills subsystem and model runner interfaces.
Uses standard library unittest.
"""

import unittest
from pathlib import Path
import sys
import os

# Add src to sys.path if not present
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from hail_core import (
    HAIL,
    HAILConfig,
    SkillManifest,
    SkillValidator,
    SkillLoader,
    HAILValidationError,
    ModelConfig,
    ModelResponse,
)

class TestSkillsAndModels(unittest.TestCase):

    def test_skill_manifest_creation(self):
        manifest = SkillManifest(
            name="test-skill",
            version="0.1.0",
            description="A test skill",
            author="Tester",
            tier="official",
            permissions=["memory:read", "memory:write"],
        )
        self.assertEqual(manifest.name, "test-skill")
        self.assertEqual(manifest.tier, "official")
        self.assertIn("memory:read", manifest.permissions)

    def test_skill_validator_success(self):
        manifest = SkillManifest(
            name="valid-skill",
            version="1.0.0",
            description="Valid skill",
            tier="community",
            permissions=["memory:read"],
        )
        is_valid, errors = SkillValidator.validate(manifest)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_skill_validator_invalid_tier(self):
        manifest = SkillManifest(
            name="invalid-tier-skill",
            version="1.0.0",
            description="Invalid tier skill",
            tier="invalid_tier",
        )
        is_valid, errors = SkillValidator.validate(manifest)
        self.assertFalse(is_valid)
        self.assertTrue(any("Invalid tier" in err for err in errors))

    def test_skill_validator_invalid_permission(self):
        manifest = SkillManifest(
            name="invalid-perm-skill",
            version="1.0.0",
            description="Invalid permission skill",
            permissions=["root:access"],
        )
        is_valid, errors = SkillValidator.validate(manifest)
        self.assertFalse(is_valid)
        self.assertTrue(any("Unrecognized permission" in err for err in errors))

    def test_skill_loader_discover(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            skills_dir = Path(tmp_dir) / "hail-skills"
            official_skill = skills_dir / "official" / "sample"
            official_skill.mkdir(parents=True)

            manifest_json = official_skill / "manifest.json"
            manifest_json.write_text("""{
                "name": "sample-skill",
                "version": "1.0.0",
                "description": "Sample skill test",
                "author": "Test Author",
                "tier": "official",
                "permissions": ["memory:read"]
            }""")

            loader = SkillLoader(skills_dir=skills_dir)
            loaded = loader.discover_and_load_all()

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].name, "sample-skill")
            self.assertIsNotNone(loader.get_skill("sample-skill"))

    def test_hail_integration_with_skills(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            skills_dir = Path(tmp_dir) / "hail-skills"
            official_skill = skills_dir / "official" / "sample"
            official_skill.mkdir(parents=True)

            manifest_json = official_skill / "manifest.json"
            manifest_json.write_text("""{
                "name": "hail-sample",
                "version": "1.0.0",
                "description": "HAIL integration test",
                "tier": "official",
                "permissions": ["memory:read", "memory:write"]
            }""")

            hail = HAIL(skills_dir=skills_dir)
            self.assertIn("hail-sample", hail.skills.list_skills())

    def test_model_config_and_response(self):
        config = ModelConfig(model_name="llama3:8b", temperature=0.5)
        response = ModelResponse(text="Hello world", tokens_used=10)
        self.assertEqual(config.model_name, "llama3:8b")
        self.assertEqual(response.text, "Hello world")

if __name__ == "__main__":
    unittest.main()
