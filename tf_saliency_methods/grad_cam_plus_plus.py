import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from tf_saliency_methods.base import SaliencyMethod
from tf_saliency_methods.utils import resize_heatmap, min_max_normalization


class GradCAMPlusPlus(SaliencyMethod):
    def __init__(self, model, last_conv_layer_name=None, output_shape=None):
        super(GradCAMPlusPlus, self).__init__(model, last_conv_layer_name)
        self.output_shape = output_shape

    def compute_cam(self, input_image: np.ndarray, pred_index: int = None) -> np.ndarray:
        """
        Generate class activation heatmap with Grad-CAM++.
        Paper: https://arxiv.org/abs/1710.11063
        """
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = Model(
            [self.model.inputs], [self.model.get_layer(
                self.last_conv_layer_name).output, self.model.output]
        )

        # Then, we compute higher-order gradients of the chosen class with regard
        # to the output feature map of the last convolutional layer.
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape3:
                    last_conv_layer_output, preds = grad_model(input_image)
                    if pred_index is None:
                        pred_index = tf.argmax(preds[0])
                    class_channel = preds[:, pred_index]
                    first_grads = tape3.gradient(
                        class_channel, last_conv_layer_output)
                second_grads = tape2.gradient(first_grads, last_conv_layer_output)
            third_grads = tape1.gradient(second_grads, last_conv_layer_output)

        global_sum = tf.reduce_sum(last_conv_layer_output,
                                   axis=(*list(range(len(input_image.shape[1:-1]))),))

        alpha_num = second_grads[0]
        alpha_denom = second_grads[0] * 2.0 + third_grads[0] * global_sum
        alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, 1e-10)

        alphas = alpha_num / alpha_denom
        alpha_normalization_constant = tf.reduce_sum(alphas, axis=(0, 1, 2))
        alphas /= alpha_normalization_constant

        weights = tf.maximum(first_grads[0], 0.0)

        deep_linearization_weights = tf.reduce_sum(
            tf.multiply(weights, alphas), axis=(*list(range(len(weights.shape))),)
        )

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = tf.reduce_sum(
            tf.multiply(deep_linearization_weights, last_conv_layer_output), axis=-1
        )

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        # Notice that we clip the heatmap values, which is equivalent to applying ReLU
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def get_cam(self, input_image: tf.Tensor, pred_index: int = None) -> np.ndarray:
        heatmap = self.compute_cam(input_image, pred_index)
        if self.output_shape is None:
            output_shape = input_image.shape[1:-1]
        else:
            output_shape = self.output_shape
        resized_heatmap = resize_heatmap(heatmap, output_shape)
        normalized_heatmap = min_max_normalization(resized_heatmap)
        return normalized_heatmap
