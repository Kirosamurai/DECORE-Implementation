"""INT8 post-training static quantization for the ONNX models.

Static PTQ quantizes weights AND activations to int8. Activations need a
calibration pass over a few hundred real images to estimate per-tensor ranges,
which is what makes conv layers (where our params live) quantizable.
"""
from __future__ import annotations

import os

import numpy as np


class CIFARCalibrationReader:
    """Feeds preprocessed CIFAR images to the ONNX static quantizer."""

    def __init__(self, num_samples: int = 200, input_name: str = "input", data_root: str = "./data"):
        import torch
        from .data import build_transforms
        import torchvision

        _, test_tf = build_transforms()
        ds = torchvision.datasets.CIFAR10(root=os.path.abspath(data_root),
                                          train=True, download=False, transform=test_tf)
        idx = list(range(min(num_samples, len(ds))))
        self._data = [ds[i][0].unsqueeze(0).numpy().astype(np.float32) for i in idx]
        self._input_name = input_name
        self._it = iter(self._data)

    def get_next(self):
        item = next(self._it, None)
        return None if item is None else {self._input_name: item}

    def rewind(self):
        self._it = iter(self._data)


def quantize_onnx_int8(fp32_path: str, int8_path: str, num_calib: int = 200,
                       data_root: str = "./data", input_name: str = "input") -> str:
    """Produce an INT8 statically-quantized ONNX model."""
    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
    from onnxruntime.quantization.shape_inference import quant_pre_process

    prepped = fp32_path.replace(".onnx", ".prep.onnx")
    try:
        quant_pre_process(fp32_path, prepped, skip_symbolic_shape=True)
    except Exception:
        prepped = fp32_path  # fall back to the raw model
    reader = CIFARCalibrationReader(num_calib, input_name, data_root)
    quantize_static(
        prepped, int8_path, reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    if prepped != fp32_path and os.path.exists(prepped):
        os.remove(prepped)
    return int8_path
