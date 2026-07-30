"""
Skill Validator for HAIL Kernel.
Ensures skills conform to security, permission, and tier standards.
"""

from typing import List, Tuple
from .manifest import SkillManifest, VALID_TIERS, VALID_PERMISSIONS
from ..exceptions import HAILValidationError

class SkillValidator:
    """Validates SkillManifest instances for safety and adherence to core rules."""

    @staticmethod
    def validate(manifest: SkillManifest) -> Tuple[bool, List[str]]:
        errors: List[str] = []

        if not manifest.name or not manifest.name.strip():
            errors.append("Skill name cannot be empty.")
        elif ".." in manifest.name or "/" in manifest.name or "\\" in manifest.name:
            errors.append(f"Invalid skill name '{manifest.name}': path traversal characters are forbidden.")

        if manifest.entrypoint and (".." in manifest.entrypoint or manifest.entrypoint.startswith("/") or manifest.entrypoint.startswith("\\")):
            errors.append(f"Invalid entrypoint '{manifest.entrypoint}': path traversal or absolute paths are forbidden.")

        if manifest.tier not in VALID_TIERS:
            errors.append(
                f"Invalid tier '{manifest.tier}'. Must be one of {sorted(list(VALID_TIERS))}."
            )

        for perm in manifest.permissions:
            if perm not in VALID_PERMISSIONS:
                errors.append(
                    f"Unrecognized permission '{perm}'. Valid permissions: {sorted(list(VALID_PERMISSIONS))}."
                )

        if not manifest.version:
            errors.append("Skill version is required.")

        is_valid = len(errors) == 0
        return is_valid, errors

    @staticmethod
    def validate_or_raise(manifest: SkillManifest) -> None:
        is_valid, errors = SkillValidator.validate(manifest)
        if not is_valid:
            raise HAILValidationError(
                f"Skill validation failed for '{manifest.name}': {'; '.join(errors)}"
            )
