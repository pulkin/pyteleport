"""
Frame object:

https://github.com/python/cpython/blob/3.8/Include/frameobject.h

Stack machine:

https://github.com/python/cpython/blob/3.8/Python/ceval.c
"""
import struct
from collections import namedtuple

from .mem_view import read_ptr, read_int, Mem


def ptr_frame_stack_bottom(data, offset=0x40):
    """Points to the bottom of frame stack"""
    return read_ptr(id(data) + offset)


def ptr_frame_stack_top(data, offset=0x48):
    """Points after the top of frame stack"""
    return read_ptr(id(data) + offset)


block_stack_item = namedtuple('block_stack_item', ('type', 'handler', 'level'))


def frame_block_stack(data, offset_size=0x70, offset_data=0x78):
    """Points after the top of frame stack"""
    size = read_int(id(data) + offset_size)
    result = struct.unpack("i" * 3 * size, Mem(id(data) + offset_data, 12 * size)[:])
    result = tuple(block_stack_item(*x) for x in zip(result[::3], result[1::3], result[2::3]))
    return result

