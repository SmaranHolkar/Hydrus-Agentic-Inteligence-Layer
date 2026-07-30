class HAILError(Exception):
    """Base exception for all HAIL memory lattice errors."""
    pass

class HAILValidationError(HAILError, ValueError):
    """Raised when validation of input parameters or embeddings fails."""
    pass

class HAILCapacityError(HAILError, RuntimeError):
    """Raised when the lattice is full and capacity limits prevent writes."""
    pass

class HAILIntegrityError(HAILError, RuntimeError):
    """Raised when cryptographic MAC checks fail (decryption key mismatch or corruption)."""
    pass
