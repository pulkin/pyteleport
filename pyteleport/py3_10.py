"""
Frame object:

https://github.com/python/cpython/blob/3.10/Include/cpython/frameobject.h

Stack machine:

https://github.com/python/cpython/blob/3.10/Python/ceval.c
"""
from . import py3_9
from .py3_9 import *
from .mem_view import read_int

JX = 2


def ptr_frame_stack_size(data, offset=0x50):
    return read_int(id(data) + offset)


def ptr_frame_stack_top(data,
    sb=ptr_frame_stack_bottom,
    ss=ptr_frame_stack_size,
    item_size=8,
):
    return sb(data) + ss(data) * item_size


def ptr_frame_block_stack_bottom(data, offset=0x70):
    return py3_9.ptr_frame_block_stack_bottom(data, offset)


def ptr_frame_block_stack_size(data, offset=0x68):
    return py3_9.ptr_frame_block_stack_size(data, offset)


def ptr_frame_block_stack_top(data,
    sb=ptr_frame_block_stack_bottom,
    ss=ptr_frame_block_stack_size,
    item_size=12,
):
    return py3_9.ptr_frame_block_stack_top(data, sb, ss, item_size)


def disassemble(arg, jx=JX, **kwargs):
    return py3_9.disassemble(arg, jx=jx, **kwargs)

