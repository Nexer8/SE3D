from setuptools import setup, find_packages

setup(
    name='torch_saliency_methods',
    version='1.0.0',
    author='Jacob Gildenblat, REDACTED',
    packages=find_packages('torch_saliency_methods, torch_saliency_methods/utils'),
    description='Many Class Activation Map methods implemented in Pytorch for classification, segmentation, object detection and more',
    long_description=open('README.md').read(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=open('requirements.txt').read().splitlines(),
)
