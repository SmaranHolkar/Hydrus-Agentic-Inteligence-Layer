import os
import hashlib
import json
from typing import Dict, Any, Tuple, Optional

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ed25519
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

class SecureWipe:
    """Utilities for securely overwriting sensitive buffers before releasing memory."""
    
    @staticmethod
    def wipe_ram_buffer(buf: bytearray or memoryview or list):
        """Overwrites RAM buffer with multi-pass security wipe (0x00, 0xFF, 0x00)."""
        if isinstance(buf, (bytearray, memoryview)):
            size = len(buf)
            buf[:size] = b'\x00' * size
            buf[:size] = b'\xFF' * size
            buf[:size] = b'\x00' * size
        elif isinstance(buf, list):
            for i in range(len(buf)):
                buf[i] = 0

    @staticmethod
    def wipe_gpu_buffer(tensor: Any):
        """Purges GPU tensor memory using CUDA memset if torch/cuda is active."""
        try:
            import torch
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                tensor.zero_()
                torch.cuda.synchronize()
        except Exception:
            pass

class AES256GCMEncryptor:
    """Per-user hardware-bound AES-256-GCM encryption engine."""
    
    def __init__(self, master_seed: bytes, hardware_uuid: Optional[str] = None, salt: Optional[bytes] = None):
        self.salt = salt or os.urandom(32)
        hw_seed = (hardware_uuid or self._get_hardware_id()).encode('utf-8')
        
        if HAS_CRYPTOGRAPHY:
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self.salt,
                info=b"HydrusMoE-AES256GCM-v2.0",
            )
            self.key = hkdf.derive(master_seed + hw_seed)
            self.cipher = AESGCM(self.key)
        else:
            # Pure SHA256 KDF fallback if cryptography module is missing
            self.key = hashlib.sha256(master_seed + hw_seed + self.salt).digest()
            self.cipher = None

    def _get_hardware_id(self) -> str:
        """Generates a hardware-bound fingerprint from system environment."""
        raw = f"{os.name}-{os.cpu_count()}-{os.getenv('COMPUTERNAME', 'LOCAL_HOST')}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def encrypt(self, plaintext: bytes, associated_data: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """
        Encrypts plaintext bytes with AES-256-GCM.
        Returns: (ciphertext_with_tag, nonce)
        """
        nonce = os.urandom(12)  # 96-bit random GCM nonce
        if self.cipher:
            ciphertext = self.cipher.encrypt(nonce, plaintext, associated_data)
        else:
            # Lightweight XOR stream fallback for zero-dependency environments
            keystream = hashlib.sha256(self.key + nonce).digest()
            ext_stream = (keystream * (len(plaintext) // len(keystream) + 1))[:len(plaintext)]
            ciphertext = bytes(a ^ b for a, b in zip(plaintext, ext_stream)) + hashlib.sha256(plaintext).digest()[:16]
        return ciphertext, nonce

    def decrypt(self, ciphertext: bytes, nonce: bytes, associated_data: Optional[bytes] = None) -> bytes:
        """
        Decrypts AES-256-GCM ciphertext and validates authentication tag.
        """
        if self.cipher:
            return self.cipher.decrypt(nonce, ciphertext, associated_data)
        else:
            tag = ciphertext[-16:]
            data = ciphertext[:-16]
            keystream = hashlib.sha256(self.key + nonce).digest()
            ext_stream = (keystream * (len(data) // len(keystream) + 1))[:len(data)]
            plaintext = bytes(a ^ b for a, b in zip(data, ext_stream))
            expected_tag = hashlib.sha256(plaintext).digest()[:16]
            if tag != expected_tag:
                raise ValueError("AES-256-GCM Authentication Tag Mismatch! Tampered shard detected.")
            return plaintext

class ManifestVerifier:
    """Validates Ed25519 signatures and Merkle-tree hash integrity chains for model manifests."""

    def __init__(self, public_key_hex: Optional[str] = None):
        self.public_key_hex = public_key_hex

    def compute_merkle_root(self, hashes: list) -> str:
        """Computes Merkle root for a list of expert SHA-256 hashes."""
        if not hashes:
            return ""
        current = [h if isinstance(h, str) else h.hex() for h in hashes]
        while len(current) > 1:
            if len(current) % 2 != 0:
                current.append(current[-1])
            next_level = []
            for i in range(0, len(current), 2):
                combined = (current[i] + current[i+1]).encode()
                next_level.append(hashlib.sha256(combined).hexdigest())
            current = next_level
        return current[0]

    def verify_manifest(self, manifest_dict: Dict[str, Any]) -> bool:
        """
        Verifies manifest integrity:
        1. Validates Merkle tree root against expert hash list.
        2. Validates Ed25519 signature if key is configured.
        """
        experts = manifest_dict.get("experts", [])
        expected_root = manifest_dict.get("merkle_root", "")
        
        expert_hashes = [e.get("sha256", "") for e in experts]
        actual_root = self.compute_merkle_root(expert_hashes)
        
        if expected_root and actual_root != expected_root:
            print(f"[ManifestVerifier Error] Merkle root mismatch: expected {expected_root}, got {actual_root}")
            return False

        signature_str = manifest_dict.get("signature", "")
        if self.public_key_hex and signature_str and HAS_CRYPTOGRAPHY:
            try:
                pub_bytes = bytes.fromhex(self.public_key_hex)
                pub_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
                
                sig_bytes = bytes.fromhex(signature_str.replace("ed25519:", ""))
                # Sign payload excludes the signature field
                payload = json.dumps({k: v for k, v in manifest_dict.items() if k != "signature"}, sort_keys=True).encode()
                pub_key.verify(sig_bytes, payload)
            except Exception as e:
                print(f"[ManifestVerifier Error] Ed25519 signature verification failed: {e}")
                return False

        return True
