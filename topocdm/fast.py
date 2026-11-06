"""
topocdm.fast -- ctypes binding for the C likelihood kernel.

Falls back to the NumPy implementation when the shared library is absent, so
the package works with or without a compiler. Build with ``python setup.py
build_ext --inplace`` (or ``scripts/build_fast.sh``, which wraps it).

Measured on the 200k x 1000 benchmark, the kernel takes the log-likelihood term
from ~41% of the fit time to a few percent (see experiments/05_kernel.py).
"""

from __future__ import annotations

import ctypes
import importlib.machinery
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _candidate_paths() -> list[str]:
    """Shared-library names to try, per platform.

    The setuptools build names the extension ``_fast<EXT_SUFFIX>`` (for
    example ``_fast.cpython-311-darwin.so`` or ``_fast.cp311-win_amd64.pyd``).
    A manual build may leave a plain ``_fast.so``/``_fast.dll``. Try both, so
    an older local build keeps working.
    """
    suffixes = list(importlib.machinery.EXTENSION_SUFFIXES)
    suffixes += [".so", ".dylib", ".dll", ".pyd"]
    names = [f"_fast{suffix}" for suffix in dict.fromkeys(suffixes)]
    return [os.path.join(_HERE, name) for name in names]


def _load_library():
    for path in _candidate_paths():
        if not os.path.exists(path):
            continue
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


_lib = _load_library()
if _lib is not None:
    try:
        _lib.chunk_loglik_csr.restype = ctypes.c_double
        _lib.chunk_loglik_csr.argtypes = [
            ctypes.POINTER(ctypes.c_long),
            ctypes.POINTER(ctypes.c_long),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_long),
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_long,
            ctypes.c_long,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        _lib.chunk_loglik_csr_g1.restype = ctypes.c_double
        _lib.chunk_loglik_csr_g1.argtypes = [
            ctypes.POINTER(ctypes.c_long),
            ctypes.POINTER(ctypes.c_long),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_long,
            ctypes.c_long,
            ctypes.POINTER(ctypes.c_double),
        ]
    except (OSError, AttributeError):
        _lib = None

AVAILABLE = _lib is not None
PLATFORM = sys.platform


def _ptr(arr, dtype):
    return arr.ctypes.data_as(ctypes.POINTER(dtype))


def chunk_loglik(X, T, u, lam, B, pi, concept_group, n_groups):
    """CSR mixture log-likelihood; returns None if the kernel is unavailable."""
    if _lib is None:
        return None
    X = X.tocsr()
    n, c = X.shape
    G = int(n_groups)
    H = int(concept_group.max()) + 1

    indptr = np.ascontiguousarray(X.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(X.indices, dtype=np.int64)
    data = np.ascontiguousarray(X.data, dtype=np.float64)
    T = np.ascontiguousarray(T, dtype=np.float64)
    u = np.ascontiguousarray(u, dtype=np.float64)
    lam = np.ascontiguousarray(lam, dtype=np.float64)
    cg = np.ascontiguousarray(concept_group, dtype=np.int64)

    if B is None or G == 1:
        lam_total = np.array([lam.sum()], dtype=np.float64)
        return float(
            _lib.chunk_loglik_csr_g1(
                _ptr(indptr, ctypes.c_long),
                _ptr(indices, ctypes.c_long),
                _ptr(data, ctypes.c_double),
                _ptr(T, ctypes.c_double),
                _ptr(u, ctypes.c_double),
                _ptr(lam, ctypes.c_double),
                ctypes.c_long(n),
                ctypes.c_long(c),
                _ptr(lam_total, ctypes.c_double),
            )
        )

    B = np.ascontiguousarray(B, dtype=np.float64)
    pi = np.ascontiguousarray(pi, dtype=np.float64)
    acc = np.zeros(G, dtype=np.float64)
    lam_h = np.zeros(H, dtype=np.float64)
    blam = np.zeros(G, dtype=np.float64)
    return float(
        _lib.chunk_loglik_csr(
            _ptr(indptr, ctypes.c_long),
            _ptr(indices, ctypes.c_long),
            _ptr(data, ctypes.c_double),
            _ptr(T, ctypes.c_double),
            _ptr(u, ctypes.c_double),
            _ptr(lam, ctypes.c_double),
            _ptr(B, ctypes.c_double),
            _ptr(pi, ctypes.c_double),
            _ptr(cg, ctypes.c_long),
            ctypes.c_long(n),
            ctypes.c_long(c),
            ctypes.c_long(G),
            ctypes.c_long(H),
            _ptr(acc, ctypes.c_double),
            _ptr(lam_h, ctypes.c_double),
            _ptr(blam, ctypes.c_double),
        )
    )
