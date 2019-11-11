from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
    name='cmr3 setup app',

    # insert name of file to compile
    ext_modules = cythonize("CMR3.pyx"),
    include_dirs=[numpy.get_include()]

)
