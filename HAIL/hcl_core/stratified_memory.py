import warnings
from hail_core.lattice import StratifiedMemoryLattice as _SML
from hail_core._internal.adaptive_strata import AdaptiveStrata as _AS
from hail_core._internal.anchored_reencoder import AnchoredReEncoder as _ARE
from hail_core._internal.hierarchical_abyss import HierarchicalAbyss as _HA
from hail_core._internal.dual_timescale_bedrock import DualTimescaleBedrock as _DTB
from hail_core._internal.temporal_binding import TemporalBinding as _TB

warnings.warn(
    "hcl_core.stratified_memory is deprecated and will be removed in a future version. "
    "Use hail_core.lattice instead.",
    DeprecationWarning,
    stacklevel=2
)

class StratifiedMemoryLattice(_SML):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "StratifiedMemoryLattice from hcl_core.stratified_memory is deprecated. "
            "Import from hail_core.lattice instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)

class AdaptiveStrata(_AS):
    pass

class AnchoredReEncoder(_ARE):
    pass

class HierarchicalAbyss(_HA):
    pass

class DualTimescaleBedrock(_DTB):
    pass

class TemporalBinding(_TB):
    pass
