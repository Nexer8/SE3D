import numpy as np
import tensorflow as tf

from tf_saliency_methods.base import SaliencyMethod


class SaliencyTubes(SaliencyMethod):
    def __init__(self, model, last_conv_layer_name=None, output_shape=None):
        super(SaliencyTubes, self).__init__(model, last_conv_layer_name, output_shape=output_shape)

    def compute_cam(self, input_image: np.ndarray, pred_index: int = None) -> np.ndarray:
        """
        Generate class activation heatmap using the Saliency Tubes method.
        Paper: https://ieeexplore.ieee.org/abstract/document/8803153
        """
        # Remove last layer's activation
        self.model.layers[-1].activation = None

        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(
                self.last_conv_layer_name).output, self.model.output]
        )

        # Then, we compute the output of the last conv layer given the input volume
        last_conv_layer_output, preds = grad_model(input_image)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])

        # We multiply each channel in the feature map array
        # by 'how important this channel is' with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        class_weights = grad_model.layers[-1].get_weights()[0]

        heatmap = np.zeros(last_conv_layer_output.shape[:-1], dtype=np.float32)
        for i, w in enumerate(class_weights[:, pred_index]):
            heatmap += w * last_conv_layer_output[..., i]

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        # Notice that we clip the heatmap values, which is equivalent to applying ReLU
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
