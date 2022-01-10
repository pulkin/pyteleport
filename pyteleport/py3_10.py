"""
Frame object:

https://github.com/python/cpython/blob/3.10/Include/cpython/frameobject.h

Stack machine:

https://github.com/python/cpython/blob/3.10/Python/ceval.c
"""
from .py3_9 import *


def stack_effect(opcode, *args, **kwargs):
    result = dis.stack_effect(opcode, *args, **kwargs)
    if opcode in (dis.opmap["GEN_START"], dis.opmap["YIELD_VALUE"]):
        return result + 1  # the actual stack effect is +1
    return result
