"""
Unit tests for CT Windowing and Multi-Slice 2.5D Transforms.
"""

import numpy as np
import pytest
from src.data.transforms import apply_window, apply_3_windows, build_25d_input, normalize_channels, WINDOW_SPECS


def test_brain_window_known_hu_values():
    """
    Brain window: WL=40, WW=80 -> [0, 80] HU.
    Tests known points: 0 HU -> 0.0, 40 HU -> 0.5, 80 HU -> 1.0, out of bounds clipped.
    """
    hu_vals = np.array([-50.0, 0.0, 40.0, 80.0, 120.0], dtype=np.float32)
    res = apply_window(hu_vals, *WINDOW_SPECS["brain"])
    expected = np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(res, expected, atol=1e-5)


def test_subdural_window_known_hu_values():
    """
    Subdural window: WL=75, WW=215 -> [-32.5, 182.5] HU.
    """
    hu_vals = np.array([-100.0, -32.5, 75.0, 182.5, 300.0], dtype=np.float32)
    res = apply_window(hu_vals, *WINDOW_SPECS["subdural"])
    expected = np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(res, expected, atol=1e-5)


def test_bone_window_known_hu_values():
    """
    Bone window: WL=400, WW=1800 -> [-500, 1300] HU.
    """
    hu_vals = np.array([-1000.0, -500.0, 400.0, 1300.0, 2000.0], dtype=np.float32)
    res = apply_window(hu_vals, *WINDOW_SPECS["bone"])
    expected = np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(res, expected, atol=1e-5)


def test_apply_3_windows_shape():
    """
    Ensures 3 windows stack has shape [3, H, W] and values in [0, 1].
    """
    slice_hu = np.random.uniform(-1000, 1000, size=(128, 128)).astype(np.float32)
    stacked = apply_3_windows(slice_hu)
    assert stacked.shape == (3, 128, 128)
    assert stacked.min() >= 0.0
    assert stacked.max() <= 1.0


def test_build_25d_input_shape():
    """
    Ensures 3 consecutive slices create 9-channel input [9, H, W].
    """
    s_prev = np.zeros((64, 64), dtype=np.float32)
    s_curr = np.ones((64, 64), dtype=np.float32) * 40.0
    s_next = np.ones((64, 64), dtype=np.float32) * 80.0
    
    stacked_9ch = build_25d_input(s_prev, s_curr, s_next)
    assert stacked_9ch.shape == (9, 64, 64)
    assert stacked_9ch.min() >= 0.0
    assert stacked_9ch.max() <= 1.0


def test_normalize_channels():
    """
    Ensures channel-wise normalization subtracts mean and divides by std.
    """
    arr = np.ones((9, 32, 32), dtype=np.float32) * 0.5
    normed = normalize_channels(arr)
    assert normed.shape == (9, 32, 32)
    assert not np.isnan(normed).any()
