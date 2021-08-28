from setuptools import setup, find_packages


setup(
    name='pyteleport',
    version='0.1.1',
    author='pulkin',
    author_email='gpulkin@gmail.com',
    packages=find_packages(),
    description='A proof-of-concept snapshot, transmission and restoring python runtime',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "dill",
    ],
)
