"""
Frame object:

https://github.com/python/cpython/blob/3.8/Include/frameobject.h

Stack machine:

https://github.com/python/cpython/blob/3.8/Python/ceval.c
"""
import struct
from collections import namedtuple
import dis

from .mem_view import read_ptr, read_int, Mem
from .minias import Bytecode

JX = 1


def ptr_frame_stack_bottom(data, offset=0x40):
    return read_ptr(id(data) + offset)


def ptr_frame_stack_top(data, offset=0x48):
    return read_ptr(id(data) + offset)


def ptr_frame_block_stack_bottom(data, offset=0x78):
    return id(data) + offset


def ptr_frame_block_stack_size(data, offset=0x70):
    return read_int(id(data) + offset)


def ptr_frame_block_stack_top(data,
    sb=ptr_frame_block_stack_bottom,
    ss=ptr_frame_block_stack_size,
    item_size=12,
):
    return sb(data) + item_size * ss(data)


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


disassemble = Bytecode.disassemble

