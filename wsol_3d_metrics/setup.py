from setuptools import setup, find_packages

setup(
    name='wsol_3d_metrics',
    version='1.0.0',
    author='Mariusz Wiśniewski (Nexer8)',
    packages=find_packages('wsol_3d_metrics'),
    # url='https://github.com/Nexer8/wsol_3d_metrics',
    description='Metrics for 3D Weakly Supervised Object Localization and Weakly Supervised Semantic Segmentation',
    long_description=open('README.md').read(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=open('requirements.txt').read().splitlines(),
)
