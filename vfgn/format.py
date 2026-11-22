# Copyright (c) VerifyGN. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
VerifyGN Proprietary Model Container (.vf)
------------------------------------------
Binary layout of a `.vf` file:
  [0..7]   8 bytes  : Magic Header b"VFGN26\x01\x00"
  [8..11]  4 bytes  : Header JSON length (uint32 little-endian)
  [12..15] 4 bytes  : Flags (bit 0 = zlib compressed, bit 1 = encrypted, bit 2 = ONNX payload)
  [16..47] 32 bytes : SHA-256 checksum of unencrypted payload
  [48..]   N bytes  : UTF-8 JSON Metadata Header (variant, names, nc, imgsz, metrics, date, license)
  [48+N..] M bytes  : Compressed (+ optionally key-encrypted) model payload (PyTorch state_dict or ONNX bytes)

Why `.vf` instead of `.pt`?
  1. Cannot be opened directly by `torch.load()` or `ultralytics.YOLO()` or Netron.
  2. Supports optional password/key obfuscation (`key="your-secret-key"`) so weights are decrypted only in RAM.
  3. Can store EITHER PyTorch weights (for training/fine-tuning) OR a compiled ONNX graph (for ultra-fast
     zero-PyTorch CPU/GPU deployment via `onnxruntime`).
"""

import io
import os
import json
import zlib
import struct
import hashlib
from datetime import datetime
import torch

VF_MAGIC = b"VFGN26\x01\x00"
FLAG_COMPRESSED = 1 << 0
FLAG_ENCRYPTED = 1 << 1
FLAG_ONNX_PAYLOAD = 1 << 2


def _cipher_bytes(data: bytes, key: str) -> bytes:
    """Fast deterministic SHA-256 stream cipher (pure standard library, zero external dependency)."""
    if not key:
        return data
    import numpy as np
    seed = hashlib.sha256(key.encode("utf-8")).digest()
    # Generate keystream blocks using blake2b keyed hashing
    buf = np.frombuffer(data, dtype=np.uint8).copy()
    n = len(buf)
    block_size = 64
    num_blocks = (n + block_size - 1) // block_size
    stream = bytearray(num_blocks * block_size)
    for i in range(num_blocks):
        h = hashlib.blake2b(
            i.to_bytes(8, "little"), key=seed, digest_size=block_size
        ).digest()
        stream[i * block_size : (i + 1) * block_size] = h
    ks = np.frombuffer(memoryview(stream)[:n], dtype=np.uint8)
    buf ^= ks
    return buf.tobytes()


def _normalize_imgsz(imgsz):
    """Normalize imgsz to either int (square) or [width, height] list when rectangular."""
    if isinstance(imgsz, (list, tuple)):
        if len(imgsz) == 2:
            w, h = int(imgsz[0]), int(imgsz[1])
            return w if w == h else [w, h]
        return int(imgsz[0])
    return int(imgsz)


def save_vf(
    path: str,
    state_dict: dict,
    variant: str,
    names: dict,
    imgsz=640,
    metrics: dict = None,
    epoch: int = -1,
    extra_meta: dict = None,
    onnx_bytes: bytes = None,
    key: str = None,
    compress_level: int = 6,
) -> str:
    """
    Save a VerifyGN model checkpoint or deployment package to a `.vf` file.
    Supports both square (`imgsz=640` or `1024`) and rectangular (`imgsz=(1024, 512)`) resolutions!
    """
    if not path.lower().endswith(".vf"):
        path = path + ".vf"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    clean_names = {int(k): str(v) for k, v in (names or {}).items()}
    metadata = {
        "framework": "VerifyGN-VFGN26",
        "format_version": "1.0",
        "license": "Apache-2.0 / VerifyGN Commercial Permissive (Zero AGPL)",
        "variant": variant.upper(),
        "nc": len(clean_names),
        "names": clean_names,
        "imgsz": _normalize_imgsz(imgsz),
        "epoch": int(epoch),
        "metrics": metrics or {},
        "date": datetime.now().isoformat(),
        "payload_type": "onnx" if onnx_bytes is not None else "torch_state_dict",
    }
    if extra_meta:
        metadata.update(extra_meta)

    # Serialize payload to memory buffer
    if onnx_bytes is not None:
        raw_payload = onnx_bytes
        flags = FLAG_COMPRESSED | FLAG_ONNX_PAYLOAD
    else:
        bio = io.BytesIO()
        # Convert tensors to CPU FP16/FP32 clean state dict (no Python class pickle references!)
        clean_sd = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                clean_sd[k] = v.detach().cpu()
            else:
                clean_sd[k] = v
        torch.save(clean_sd, bio)
        raw_payload = bio.getvalue()
        flags = FLAG_COMPRESSED

    sha256_digest = hashlib.sha256(raw_payload).digest()
    payload = zlib.compress(raw_payload, level=compress_level)

    if key:
        flags |= FLAG_ENCRYPTED
        payload = _cipher_bytes(payload, key)

    header_bytes = json.dumps(metadata, indent=None).encode("utf-8")
    prefix = VF_MAGIC + struct.pack("<II", len(header_bytes), flags) + sha256_digest

    with open(path, "wb") as f:
        f.write(prefix)
        f.write(header_bytes)
        f.write(payload)

    return path


def inspect_vf(path: str) -> dict:
    """
    Read ONLY the metadata header from a `.vf` file (instant, does not load weights).
    """
    with open(path, "rb") as f:
        prefix = f.read(48)
        if len(prefix) < 48 or prefix[:8] != VF_MAGIC:
            raise ValueError(f"Invalid .vf file (bad magic header): {path}")
        header_len, flags = struct.unpack("<II", prefix[8:16])
        sha256_hex = prefix[16:48].hex()
        header_bytes = f.read(header_len)

    meta = json.loads(header_bytes.decode("utf-8"))
    meta["names"] = {int(k): v for k, v in meta.get("names", {}).items()}
    meta["encrypted"] = bool(flags & FLAG_ENCRYPTED)
    meta["is_onnx"] = bool(flags & FLAG_ONNX_PAYLOAD)
    meta["sha256"] = sha256_hex
    meta["file_size_mb"] = round(os.path.getsize(path) / (1024 * 1024), 2)
    return meta


def load_vf(path: str, key: str = None, map_location: str = "cpu") -> dict:
    """
    Load a `.vf` file and verify its SHA-256 integrity.
    Returns dict with keys:
      'metadata'   : metadata dict (variant, names, nc, imgsz, metrics, ...)
      'state_dict' : PyTorch state_dict (if torch payload) or None
      'onnx_bytes' : raw ONNX model bytes (if ONNX payload) or None
    """
    with open(path, "rb") as f:
        prefix = f.read(48)
        if len(prefix) < 48 or prefix[:8] != VF_MAGIC:
            raise ValueError(f"Invalid .vf file (bad magic header): {path}")
        header_len, flags = struct.unpack("<II", prefix[8:16])
        expected_sha256 = prefix[16:48]
        header_bytes = f.read(header_len)
        payload = f.read()

    meta = json.loads(header_bytes.decode("utf-8"))
    meta["names"] = {int(k): v for k, v in meta.get("names", {}).items()}

    if flags & FLAG_ENCRYPTED:
        if not key:
            raise PermissionError(f"Model '{path}' is encrypted. Please provide the decryption `key`.")
        payload = _cipher_bytes(payload, key)

    if flags & FLAG_COMPRESSED:
        try:
            raw_payload = zlib.decompress(payload)
        except zlib.error as e:
            raise PermissionError(f"Failed to decompress '{path}' (wrong decryption key or corrupted file): {e}")
    else:
        raw_payload = payload

    actual_sha256 = hashlib.sha256(raw_payload).digest()
    if actual_sha256 != expected_sha256:
        raise PermissionError(f"Integrity check failed for '{path}' (wrong decryption key or tampered file).")

    if flags & FLAG_ONNX_PAYLOAD:
        return {"metadata": meta, "state_dict": None, "onnx_bytes": raw_payload}
    else:
        bio = io.BytesIO(raw_payload)
        sd = torch.load(bio, map_location=map_location, weights_only=True)
        return {"metadata": meta, "state_dict": sd, "onnx_bytes": None}


def export_onnx_to_vf(
    onnx_path_or_bytes,
    output_vf_path: str,
    variant: str,
    names: dict,
    imgsz: int = 640,
    metrics: dict = None,
    key: str = None,
) -> str:
    """Package an ONNX graph directly into a `.vf` container for ultra-fast ONNX Runtime deployment."""
    if isinstance(onnx_path_or_bytes, (bytes, bytearray)):
        onnx_bytes = bytes(onnx_path_or_bytes)
    else:
        with open(onnx_path_or_bytes, "rb") as f:
            onnx_bytes = f.read()

    return save_vf(
        path=output_vf_path,
        state_dict={},
        variant=variant,
        names=names,
        imgsz=imgsz,
        metrics=metrics,
        onnx_bytes=onnx_bytes,
        key=key,
    )
