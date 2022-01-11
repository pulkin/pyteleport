"""
Frame object:

https://github.com/python/cpython/blob/3.8/Include/frameobject.h

Stack machine:

https://github.com/python/cpython/blob/3.8/Python/ceval.c
"""
import dis

from .util import lookup_nested


interrupting = lookup_nested(dis.opmap, (
    "JUMP_ABSOLUTE",
    "JUMP_FORWARD",
    "RETURN_VALUE",
    "RAISE_VARARGS",
))


stack_effect = dis.stack_effect
