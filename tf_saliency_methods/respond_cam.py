import numpy as np
import tensorflow as tf

from tf_saliency_methods.base import SaliencyMethod


class RespondCAM(SaliencyMethod):
    def __init__(self, model, last_conv_layer_name=None, output_shape=None):
        super(RespondCAM, self).__init__(model, last_conv_layer_name, output_shape=output_shape)

    def compute_cam(self, input_image: np.ndarray, pred_index: int = None) -> np.ndarray:
        """
        Generate class activation heatmap with HiResCAM.
        Paper: https://arxiv.org/abs/2011.08891
        """
        # Remove last layer's activation
        self.model.layers[-1].activation = None

        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
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

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class,
        # then sum all the channels to obtain the heatmap class activation.
        last_conv_layer_output = last_conv_layer_output[0]

        # Respond-CAM is fairly similar to HiResCAM but it uses weighted,
        # instead of raw, feature maps
        respond_weights = np.sum(last_conv_layer_output * grads, axis=(*list(range(len(last_conv_layer_output.shape) - 1)),)) / \
            (np.sum(last_conv_layer_output + 1e-10, (*list(range(len(last_conv_layer_output.shape) - 1)),)))

        heatmap = last_conv_layer_output * respond_weights
        heatmap = np.sum(heatmap, axis=-1)
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        # Notice that we clip the heatmap values, which is equivalent to applying ReLU
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
