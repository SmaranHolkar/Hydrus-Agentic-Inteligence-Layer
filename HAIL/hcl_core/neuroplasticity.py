import warnings
from hail_core.neuroplasticity import InstantNeuroplasticity as _IN

warnings.warn(
    "hcl_core.neuroplasticity is deprecated and will be removed in a future version. "
    "Use hail_core.neuroplasticity instead.",
    DeprecationWarning,
    stacklevel=2
)

class InstantNeuroplasticity(_IN):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "InstantNeuroplasticity from hcl_core.neuroplasticity is deprecated. "
            "Import from hail_core.neuroplasticity instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
