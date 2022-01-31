from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize


setup(
    name='pyteleport',
    version='0.2',
    author='pulkin',
    author_email='gpulkin@gmail.com',
    packages=find_packages(),
    description='A proof-of-concept snapshot, transmission and restoring python runtime',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    ext_modules=cythonize([
        Extension("pyteleport.frame", ["cython/frame.pyx"]),
    ]),
    install_requires=[
        "dill",
        "Cython",
    ],
)
