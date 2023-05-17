from typing import Tuple

import numpy as np
import tensorflow as tf

from tf_saliency_methods.utils import resize_heatmap, min_max_normalization


class SaliencyMethod(object):
    def __init__(self, model: tf.keras.Model, last_conv_layer_name: str = None, output_shape: Tuple = None):
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        self.output_shape = output_shape

    def compute_cam(self, input_image, pred_index: int = None):
        raise NotImplementedError('Must be implemented by subclass.')

    def get_cam(self, input_image: np.ndarray, pred_index: int = None) -> np.ndarray:
        heatmap = self.compute_cam(input_image, pred_index)
        if self.output_shape is None:
            output_shape = input_image.shape[1:-1]
        else:
            output_shape = self.output_shape

        if output_shape != heatmap.shape:
            heatmap = resize_heatmap(heatmap, output_shape)
        normalized_heatmap = min_max_normalization(heatmap)
        return normalized_heatmap
