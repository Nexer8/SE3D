# TF Saliency Methods

Author: `REDACTED` *(available upon publication)*

This repository contains implementations of saliency methods for TensorFlow 2.0. The methods are implemented as Keras
callbacks, and can be used with any Keras model for input data of any dimensionality.

## Table of Contents

- [Implemented Methods](#implemented-methods)
- [Usage](#usage)
    - [Requirements](#requirements)
    - [Code Structure](#code-structure)
    - [Example](#example)
- [Code License](#code-license)

## Implemented Methods

- [Grad-CAM](https://arxiv.org/abs/1610.02391): weights the gradients of the target class with the activations of the last convolutional layer.
- [Grad-CAM++](https://arxiv.org/abs/1710.11063): extends Grad-CAM by adding higher-order derivatives.
- [HiResCAM](https://arxiv.org/abs/2011.08891): replaces the global average pooling layer with element-wise multiplication of the feature maps and the gradients.
- [Respondd-CAM](https://arxiv.org/abs/1806.00102): 3D-specific saliency method, which uses weighted-average of all the gradients in the feature map to smoothen the gradients.
- [Saliency Tubes](https://arxiv.org/abs/1902.01078): 3D-specific saliency method that computes the saliency map by multiplying the class weights with the feature maps of the last convolutional layer.

## Usage

To use the metrics in your project, you can install this repository as a pip package:

```sh
pip install -e tf_saliency_methods
```

### Requirements

The requirements for this repository are listed in `requirements.txt`. In case you want to install the package manually,
you can do so by running:

```sh
pip install -r requirements.txt
```

### Code Structure

The code is structured as follows:

```sh
tf_saliency_methods             # root directory
├── README.md                   # this file
├── __init__.py                 # package initialization
├── base.py                     # base class for all saliency methods
├── examples                    # example notebooks
│   └── grad_cam_example.ipynb  # example notebook for Grad-CAM
├── grad_cam.py                 # implements Grad-CAM
├── grad_cam_plus_plus.py       # implements Grad-CAM++
├── hirescam.py                 # implements HiResCAM
├── requirements.txt            # requirements for this repository
├── respond_cam.py              # implements Respond-CAM
├── saliency_tubes.py           # implements Saliency Tubes
├── setup.py                    # setup file for pip installation
└── utils                       # utility functions
    ├── __init__.py             # initialization
    ├── data_processing.py      # data processing functions
    └── find_layers.py          # functions to find layers in a model
```

### Example

The following example shows how to use the saliency methods in your project.

```python
import tensorflow as tf
from tf_saliency_methods import GradCAM

# define last convolutional layer name
last_conv_layer_name = "block5_conv3"
# define class index
pred_index = 1

# load model
model = tf.keras.load_model("path/to/model.h5")

# load image
img = tf.keras.preprocessing.image.load_img("path/to/image.jpg", target_size=(224, 224))

# create saliency method
grad_cam = GradCAM(model, last_conv_layer_name=last_conv_layer_name)

# compute saliency map
heatmap = grad_cam.get_cam(img, pred_index=pred_index)
```

To see more examples, please refer to the `examples/` directory.

## Code License

To be determined.
