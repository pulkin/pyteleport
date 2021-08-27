"""
Frame object:

https://github.com/python/cpython/blob/3.8/Include/frameobject.h

Stack machine:

https://github.com/python/cpython/blob/3.8/Python/ceval.c
"""
from .mem_view import read_ptr


def ptr_frame_stack_bottom(data):
    """Points to the bottom of frame stack"""
    return read_ptr(id(data) + 0x40)


def ptr_frame_stack_top(data):
    """Points after the top of frame stack"""
    return read_ptr(id(data) + 0x48)

