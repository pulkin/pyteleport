from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize


setup(
    packages=find_packages(),
    ext_modules=cythonize([
        Extension("pyteleport.frame", ["cython/frame.pyx"]),
    ]),
)
