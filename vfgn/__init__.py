# Copyright (c) VerifyGN. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
VFGN26 Proprietary Object Detection Framework
Variants : VFGN26S (Small), VFGN26M (Medium), VFGN26XL (Extra-Large)
Format   : .vf (VerifyGN Binary Model Container with optional AES/XOR stream cipher)
License  : 100% Apache-2.0 / MIT / BSD-3 stack (Zero Ultralytics / Zero AGPL-3.0)
"""

from .model import VFGNModel, VFGN26Detector, VARIANT_CONFIGS
from .format import save_vf, load_vf, inspect_vf, export_onnx_to_vf

__version__ = "1.0.0"
__all__ = [
    "VFGNModel",
    "VFGN26Detector",
    "VARIANT_CONFIGS",
    "save_vf",
    "load_vf",
    "inspect_vf",
    "export_onnx_to_vf",
]
