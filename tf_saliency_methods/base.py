class SaliencyMethod(object):
    def __init__(self, model, last_conv_layer_name=None):
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name

    def get_cam(self, input_image):
        raise NotImplementedError('Must be implemented by subclass.')
