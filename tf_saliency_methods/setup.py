from setuptools import setup, find_packages

setup(
    name='tf_saliency_methods',
    version='1.0.0',
    author='REDACTED',
    packages=find_packages('tf_saliency_methods', 'tf_saliency_methods/utils'),
    # url='REDACTED',
    description='Tensorflow 2.0 implementation of popular Saliency Methods',
    long_description=open('README.md').read(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=open('requirements.txt').read().splitlines(),
)
