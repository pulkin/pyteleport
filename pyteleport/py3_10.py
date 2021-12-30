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


@dataclass
class ExtendedFrameInfo(py3_9.ExtendedFrameInfo):
    o_stack_size: int = 0x50
    o_bstack_bottom: int = 0x70
    o_bstack_size: int = 0x68

    @property
    def a_stack_size(self):
        return self.fid + self.o_stack_size

    @property
    def a_stack_top(self):
        """Changed in 3.10: stack top pointer is replaced by the stack size"""
        raise NotImplementedError

    def get_stack_size(self):
        return read_int(self.a_stack_size)

    def ptr_frame_stack_top(self, item_size=8):
        return self.ptr_frame_stack_bottom() + item_size * self.get_stack_size()
