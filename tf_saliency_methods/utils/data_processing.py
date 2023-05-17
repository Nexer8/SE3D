from typing import Tuple

import numpy as np
from scipy.ndimage import zoom


def resize_heatmap(heatmap: np.ndarray, shape: Tuple) -> np.ndarray:
    """Resize heatmap to shape"""
    assert len(heatmap.shape) == len(shape), \
        "Heatmap and shape must have the same number of dimensions"

    return zoom(
        heatmap,
        tuple(shape[i] / heatmap.shape[i] for i in range(len(shape))),
    )


def min_max_normalization(x: np.ndarray) -> np.ndarray:
    """Min-max normalization"""
    return (x - np.min(x)) / (np.max(x) - np.min(x))
