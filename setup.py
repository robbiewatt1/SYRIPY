from setuptools import setup

setup(
    name='ERS',
    version='1.0',
    description='A package for modelling the generation and propagation of '
                'edge radiation ',
    author='Robbie Watt',
    author_email='rwatt1@stanford.edu',
    packages=['ERS'],
    package_dir={'ERS': 'source'},
    install_requires=[
        'numpy',
        'matplotlib',
        'torch'
    ],
)