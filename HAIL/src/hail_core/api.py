import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

from .lattice import StratifiedMemoryLattice
from .skills import SkillLoader, SkillManifest

@dataclass
class HAILConfig:
    lattice_size: int = 65536
    dim: int = 64
    storage_path: Optional[Path] = None   # None = in-memory only, no persistence
    skills_dir: Optional[Path] = None     # Path to hail-skills directory
    passphrase: Optional[str] = None      # None = unencrypted local file
    autosave: bool = False                # if True, save on __exit__
    thread_safe: bool = True

class HAIL:
    """
    High-level entry point for the HAIL Core SDK.
    Wraps StratifiedMemoryLattice with ergonomic configuration,
    skill management, context-manager support, thread-safety, and local persistence.
    """

    def __init__(self, config: Optional[HAILConfig] = None, **kwargs):
        if config is None:
            self.config = HAILConfig(**kwargs)
        else:
            self.config = config

        lock = threading.RLock() if self.config.thread_safe else None
        self._lattice = StratifiedMemoryLattice(
            lattice_size=self.config.lattice_size,
            dim=self.config.dim,
            lock=lock
        )
        self.skills = SkillLoader(skills_dir=self.config.skills_dir)

        if self.config.skills_dir and Path(self.config.skills_dir).exists():
            self.skills.discover_and_load_all()

        if self.config.storage_path:
            # Convert storage_path to Path object to ensure type safety
            self.config.storage_path = Path(self.config.storage_path)
            if self.config.storage_path.exists():
                self._load()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.config.autosave and self.config.storage_path:
            self.save()
        return False  # Do not suppress exceptions

    @property
    def lattice(self) -> StratifiedMemoryLattice:
        """Access the underlying StratifiedMemoryLattice engine directly."""
        return self._lattice

    def save(self):
        """Save the memory lattice to disk. Raises ValueError if storage_path is not set."""
        if not self.config.storage_path:
            raise ValueError("No storage_path configured — set one in HAILConfig to persist.")
        self._lattice.save_to_disk(str(self.config.storage_path), self.config.passphrase)

    def _load(self):
        """Load the memory lattice from disk."""
        if not self.config.storage_path:
            return
        self._lattice.load_from_disk(str(self.config.storage_path), self.config.passphrase)

    def write(self, embedding: Any, confidence: float = 0.8, payload: Optional[Dict] = None) -> int:
        """Write an embedding fact to the lattice with a given confidence and metadata payload."""
        return self._lattice.write(embedding, confidence, payload)

    def recall(self, query_embedding: Any, k: int = 5, user_confirmed: bool = False) -> List[Dict[str, Any]]:
        """Recall top-k matches for a query embedding, updating thermal weights and strata context."""
        return self._lattice.recall(query_embedding, k, user_confirmed)

    def forget_to_abyss(self, addr: int):
        """Evict a memory address from the surface and archive it in the Hierarchical Abyss."""
        self._lattice.forget_to_abyss(addr)

    def thermal_decay(self, lambda_: float = 0.05) -> Dict[int, int]:
        """Apply passive thermal decay to all occupied memories and trigger any needed migrations."""
        return self._lattice.thermal_decay(lambda_)

    def compute_abstraction_score(self, addr: int) -> float:
        """Compute the query context-spread based abstraction score for a memory slot."""
        return self._lattice.compute_abstraction_score(addr)
