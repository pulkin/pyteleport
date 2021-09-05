"""
Frame object:

https://github.com/python/cpython/blob/3.9/Include/cpython/frameobject.h

Stack machine:

https://github.com/python/cpython/blob/3.9/Python/ceval.c
"""
from .py3_8 import *


def put_NULL(code):
    """
    BEGIN_FINALLY is gone
    """
    raise NotImplementedError("NULLs on stack are not implemented")

