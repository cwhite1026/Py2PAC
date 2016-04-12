#My attempt at a setup... I have no idea how this is supposed to work.

from setuptools import setup
import os
import sys

#Compile the cython pieces with the proper options
sys.argv = ['setup.py', 'build_ext', '--inplace']
import distutils.core as core
from Cython.Build import cythonize
import numpy
core.setup(
    ext_modules = cythonize("py2pac/image_mask_cybits.pyx"),
    include_dirs=[numpy.get_include()]
)

#Define the readme command
def readme():
    with open('README.md') as f:
        return f.read()

#Set up the package
setup(name='py2pac',
      version='0.1',
      description='Python package for computing 2-point angular correlation functions',
      url='https://github.com/cwhite1026/Py2PAC',
      install_requires = ['numpy>=1.10', "cython>=0.24"],
      author='Catherine White',
      author_email='ccavigl1@jh.edu',
      packages=['py2pac'],
      package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst']
        },
      license = "BSD",
      )
