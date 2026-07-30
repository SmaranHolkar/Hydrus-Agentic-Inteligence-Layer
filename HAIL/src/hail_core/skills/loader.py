"""
Skill Loader & Registry for HAIL Kernel.
Loads, validates, and registers skills into kernel state.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from .manifest import SkillManifest
from .validator import SkillValidator
from ..exceptions import HAILValidationError

class SkillLoader:
    """Discovers, validates, and manages active skills in the kernel."""

    def __init__(self, skills_dir: Optional[Path] = None):
        self.skills_dir = skills_dir
        self.loaded_skills: Dict[str, SkillManifest] = {}

    def load_from_dict(self, data: Dict) -> SkillManifest:
        manifest = SkillManifest.from_dict(data)
        SkillValidator.validate_or_raise(manifest)
        self.loaded_skills[manifest.name] = manifest
        return manifest

    def load_from_path(self, skill_path: Path) -> SkillManifest:
        skill_path = Path(skill_path)
        manifest_file = skill_path / "manifest.json"

        if not manifest_file.exists():
            # Fallback to checking for SKILL.md frontmatter or simple folder name
            manifest_file = skill_path / "skill.json"

        if not manifest_file.exists():
            raise HAILValidationError(f"No skill manifest found at '{skill_path}'")

        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self.load_from_dict(data)

    def discover_and_load_all(self, base_dir: Optional[Path] = None) -> List[SkillManifest]:
        target_dir = base_dir or self.skills_dir
        if not target_dir or not Path(target_dir).exists():
            return []

        loaded: List[SkillManifest] = []
        target_dir = Path(target_dir)

        # Iterate through tiers: official, community, experimental, or direct subfolders
        for path in target_dir.rglob("manifest.json"):
            try:
                manifest = self.load_from_path(path.parent)
                loaded.append(manifest)
            except Exception as e:
                # Log or skip invalid skills gracefully
                continue

        return loaded

    def get_skill(self, name: str) -> Optional[SkillManifest]:
        return self.loaded_skills.get(name)

    def list_skills(self) -> List[str]:
        return list(self.loaded_skills.keys())
