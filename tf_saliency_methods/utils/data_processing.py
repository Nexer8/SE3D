from typing import Tuple

import numpy as np
from scipy.ndimage import zoom


def resize_heatmap(heatmap: np.ndarray, shape: Tuple) -> np.ndarray:
    """Resize heatmap to shape"""
    assert len(heatmap.shape) == len(shape), \
        "Heatmap and shape must have the same number of dimensions"
    assert heatmap.shape[-1] == 1, \
        "Heatmap must have a single channel"

    return zoom(
        heatmap,
        (shape[i] / heatmap.shape[i] for i in range(len(shape) - 1)),
    )


def min_max_normalization(x: np.ndarray) -> np.ndarray:
    """Min-max normalization"""
    return (x - np.min(x)) / (np.max(x) - np.min(x))
