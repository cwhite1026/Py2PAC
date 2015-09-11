from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("image_mask_cybits.pyx"),
    include_dirs=[numpy.get_include()]
)
