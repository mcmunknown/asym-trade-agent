"""
C++ bridge for high-performance calculus kernels.

Loads the compiled libmathcore shared library and exposes NumPy-friendly
wrappers that mirror the legacy Python implementations.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

LIB_PATH = Path(__file__).resolve().parent.parent / "cpp" / "libmathcore.dylib"

_lib: Optional[ctypes.CDLL] = None
if LIB_PATH.exists():
    try:
        _lib = ctypes.CDLL(str(LIB_PATH))
    except OSError as exc:
        raise RuntimeError(f"Failed to load math core library at {LIB_PATH}: {exc}") from exc

_double_ptr = ctypes.POINTER(ctypes.c_double)

if _lib is not None:
    _lib.mc_exponential_smoothing.argtypes = [
        _double_ptr,
        ctypes.c_size_t,
        ctypes.c_double,
        _double_ptr,
    ]
    _lib.mc_exponential_smoothing.restype = None

    _lib.mc_velocity.argtypes = [
        _double_ptr,
        ctypes.c_size_t,
        ctypes.c_double,
        _double_ptr,
    ]
    _lib.mc_velocity.restype = None

    _lib.mc_acceleration.argtypes = [
        _double_ptr,
        ctypes.c_size_t,
        ctypes.c_double,
        _double_ptr,
    ]
    _lib.mc_acceleration.restype = None

    _lib.mc_analyze_curve.argtypes = [
        _double_ptr,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
        _double_ptr,
        _double_ptr,
        _double_ptr,
    ]
    _lib.mc_analyze_curve.restype = None


def cpp_available() -> bool:
    """Return True if the compiled library is available."""
    return _lib is not None


def _as_array_and_ptr(series: np.ndarray) -> Tuple[np.ndarray, ctypes.Array]:
    array = np.asarray(series, dtype=np.float64, order="C")
    return array, array.ctypes.data_as(_double_ptr)


def exponential_smoothing(prices: np.ndarray, lambda_param: float) -> np.ndarray:
    if _lib is None:
        raise RuntimeError("math core library not loaded")
    prices_arr, prices_ptr = _as_array_and_ptr(prices)
    out = np.empty_like(prices_arr)
    out_ptr = out.ctypes.data_as(_double_ptr)
    _lib.mc_exponential_smoothing(
        prices_ptr,
        prices_arr.shape[0],
        float(lambda_param),
        out_ptr,
    )
    return out


def velocity(smoothed: np.ndarray, dt: float) -> np.ndarray:
    if _lib is None:
        raise RuntimeError("math core library not loaded")
    smoothed_arr, smoothed_ptr = _as_array_and_ptr(smoothed)
    out = np.empty_like(smoothed_arr)
    out_ptr = out.ctypes.data_as(_double_ptr)
    _lib.mc_velocity(
        smoothed_ptr,
        smoothed_arr.shape[0],
        float(dt),
        out_ptr,
    )
    return out


def acceleration(velocity_series: np.ndarray, dt: float) -> np.ndarray:
    if _lib is None:
        raise RuntimeError("math core library not loaded")
    velocity_arr, velocity_ptr = _as_array_and_ptr(velocity_series)
    out = np.empty_like(velocity_arr)
    out_ptr = out.ctypes.data_as(_double_ptr)
    _lib.mc_acceleration(
        velocity_ptr,
        velocity_arr.shape[0],
        float(dt),
        out_ptr,
    )
    return out


def analyze_curve(prices: np.ndarray,
                  lambda_param: float,
                  dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _lib is None:
        raise RuntimeError("math core library not loaded")
    prices_arr, prices_ptr = _as_array_and_ptr(prices)
    smoothed = np.empty_like(prices_arr)
    velocity_out = np.empty_like(prices_arr)
    acceleration_out = np.empty_like(prices_arr)
    _lib.mc_analyze_curve(
        prices_ptr,
        prices_arr.shape[0],
        float(lambda_param),
        float(dt),
        smoothed.ctypes.data_as(_double_ptr),
        velocity_out.ctypes.data_as(_double_ptr),
        acceleration_out.ctypes.data_as(_double_ptr),
    )
    return smoothed, velocity_out, acceleration_out
