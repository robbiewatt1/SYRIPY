from skbuild import setup
from setuptools import find_packages
from setuptools.command.install import install


setup(
    name='SYRIPY',
    version='1.0',
    description='A package for modelling the generation and propagation of '
                'edge radiation ',
    author='Robbie Watt',
    author_email='rwatt1@stanford.edu',
    packages=find_packages(),
    cmake_install_dir="SYRIPY/Tracking/cTrack",
    install_requires=[
        'numpy',
        'matplotlib',
        'torch'
    ],
)