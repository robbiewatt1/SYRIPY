import os
from skbuild import setup
from setuptools import find_packages
from setuptools.command.install import install

def set_cmake_args():
    args = os.environ.get('BUILD_LIBTORCH', None)
    if args :
        return ["-DBUILD_LIBTORCH=ON"]
    else:
        return ["-DBUILD_LIBTORCH=OFF"]


setup(
    name='SYRIPY',
    version='1.0',
    description='A package for modelling the generation and propagation of '
                'synchrotron radiation ',
    author='Robbie Watt',
    author_email='rwatt1@stanford.edu',
    packages=find_packages(),
    cmake_args=set_cmake_args(),
    cmake_install_dir="SYRIPY/Tracking/cTrack",
    install_requires=[
        'numpy',
        'matplotlib',
        'torch'])
