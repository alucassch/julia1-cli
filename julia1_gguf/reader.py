"""GGUF access: metadata as plain Python values, tensors as float32 arrays in PyTorch [out, in] layout.

GGUF stores a PyTorch [out, in] matrix with ne = [in, out]; `GGUFReader` already
exposes `tensor.data` with the rows-first (numpy) shape, i.e. [out, bytes_per_row]
for quantised types and [out, in] for F32/F16, so `gguf.quants.dequantize`
returns [out, in] directly (checked for F32, F16, BF16, Q8_0, Q5_0, Q4_0).
"""
from __future__ import annotations

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


class GGUFModel:
    def __init__(self, path):
        self.path = str(path)
        self.reader = GGUFReader(self.path)
        self.metadata = {name: field.contents() for name, field in self.reader.fields.items()
                         if not name.startswith('GGUF.')}
        self._tensors = {t.name: t for t in self.reader.tensors}

    def __getitem__(self, key):
        return self.metadata[key]

    def get(self, key, default=None):
        return self.metadata.get(key, default)

    @property
    def tensor_names(self):
        return list(self._tensors)

    def tensor_type(self, name):
        return self._tensors[name].tensor_type

    def tensor(self, name) -> np.ndarray:
        """Whole tensor dequantised to float32, PyTorch-shaped [out, in] (1-D tensors unchanged)."""
        t = self._tensors[name]
        return np.ascontiguousarray(dequantize(np.asarray(t.data), t.tensor_type), dtype=np.float32)

    def rows(self, name, ids) -> np.ndarray:
        """Rows `ids` of a 2-D tensor, dequantised to float32 [len(ids), in] without touching the rest."""
        t = self._tensors[name]
        picked = np.ascontiguousarray(t.data[np.asarray(ids, dtype=np.int64)])
        return np.ascontiguousarray(dequantize(picked, t.tensor_type), dtype=np.float32)
