import unittest
import os
import sys
from pathlib import Path

# Add HAIL to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "HAIL"))

from hydrusmoe.crypto import AES256GCMEncryptor, ManifestVerifier, SecureWipe

class TestMoECrypto(unittest.TestCase):

    def setUp(self):
        self.master_seed = os.urandom(32)
        self.encryptor = AES256GCMEncryptor(self.master_seed, hardware_uuid="TEST-HW-12345")

    def test_aes_gcm_roundtrip(self):
        plaintext = b"HydrusMoE_Quantized_Expert_Weights_Shard_001"
        ciphertext, nonce = self.encryptor.encrypt(plaintext)
        decrypted = self.encryptor.decrypt(ciphertext, nonce)
        self.assertEqual(plaintext, decrypted)

    def test_hardware_binding_integrity(self):
        plaintext = b"Sensitive_Expert_Weights"
        ciphertext, nonce = self.encryptor.encrypt(plaintext)
        
        # Instantiate encryptor with different hardware UUID
        other_encryptor = AES256GCMEncryptor(self.master_seed, hardware_uuid="OTHER-HW-99999", salt=self.encryptor.salt)
        
        with self.assertRaises(Exception):
            other_encryptor.decrypt(ciphertext, nonce)

    def test_manifest_verification(self):
        verifier = ManifestVerifier()
        hashes = ["a" * 64, "b" * 64, "c" * 64]
        merkle_root = verifier.compute_merkle_root(hashes)
        
        manifest = {
            "model_id": "qwen3-35b-a3b",
            "version": "1.0.2",
            "merkle_root": merkle_root,
            "experts": [{"id": 0, "sha256": hashes[0]}, {"id": 1, "sha256": hashes[1]}, {"id": 2, "sha256": hashes[2]}]
        }
        self.assertTrue(verifier.verify_manifest(manifest))

    def test_secure_wipe(self):
        buf = bytearray(b"SECRET_WEIGHTS_BUFFER")
        SecureWipe.wipe_ram_buffer(buf)
        self.assertEqual(buf, bytearray(b"\x00" * 21))

if __name__ == "__main__":
    unittest.main()
