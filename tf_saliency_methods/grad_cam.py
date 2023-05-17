import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from tf_saliency_methods.base import SaliencyMethod


class GradCAM(SaliencyMethod):
    def __init__(self, model, last_conv_layer_name=None, output_shape=None):
        super(GradCAM, self).__init__(model, last_conv_layer_name, output_shape=output_shape)

    def compute_cam(self, input_image: np.ndarray, pred_index: int = None) -> np.ndarray:
        """
        Generate class activation heatmap with Grad-CAM.
        Paper: https://arxiv.org/abs/1610.02391
        Code: https://keras.io/examples/vision/grad_cam/
        """
        # Remove last layer's activation
        self.model.layers[-1].activation = None

        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions

        grad_model = Model(
            [self.model.inputs], [self.model.get_layer(
                self.last_conv_layer_name).output, self.model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(input_image)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel (equivalent to global average pooling)
        pooled_grads = tf.reduce_mean(
            grads, axis=(*list(range(len(last_conv_layer_output.shape) - 1)),)
        )

        # We multiply each channel in the feature map array
        # by 'how important this channel is' with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        # Notice that we clip the heatmap values, which is equivalent to applying ReLU
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
