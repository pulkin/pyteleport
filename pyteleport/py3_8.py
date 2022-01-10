"""
Frame object:

https://github.com/python/cpython/blob/3.8/Include/frameobject.h

Stack machine:

https://github.com/python/cpython/blob/3.8/Python/ceval.c
"""
import dis
from collections import namedtuple

from .util import lookup_nested


interrupting = lookup_nested(dis.opmap, (
    "JUMP_ABSOLUTE",
    "JUMP_FORWARD",
    "RETURN_VALUE",
    "RAISE_VARARGS",
))
block_edges = namedtuple("block_edges", ("start", "end"))


def put_NULL(code):
    """Simply puts NULL on the stack."""
    code.i(dis.opmap["BEGIN_FINALLY"], 0)


def put_EXCEPT_HANDLER(code):
    """Puts except handler and 3 items (NULL, NULL, None) on the stack"""
    setup_finally = code.I(dis.opmap["SETUP_FINALLY"], None)
    code.i(dis.opmap["RAISE_VARARGS"], 0)
    for i in range(3):
        pop_top = code.i(dis.opmap["POP_TOP"], 0)
        if i == 0:
            setup_finally.jump_to = pop_top


stack_effect = dis.stack_effect
