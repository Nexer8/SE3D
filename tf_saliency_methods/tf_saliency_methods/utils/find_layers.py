import tensorflow as tf


def find_last_layer_by_type(layer_type: type, model: tf.keras.Model) -> str:
    """
    Find the last layer of a given type in a model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, layer_type):
            return layer.name
    raise ValueError(f'No layer of type {layer_type} found in model.')


def find_last_layer_by_name(layer_name: str, model: tf.keras.Model) -> str:
    """
    Find the last layer of a given name in a model.
    """
    for layer in reversed(model.layers):
        if layer.name.startswith(layer_name):
            return layer.name
    raise ValueError(f'No layer named {layer_name} found in model.')
