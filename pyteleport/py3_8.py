"""
Frame object:

https://github.com/python/cpython/blob/3.8/Include/frameobject.h

Stack machine:

https://github.com/python/cpython/blob/3.8/Python/ceval.c
"""
import struct
from dataclasses import dataclass, field
import dis
from types import FrameType

from .mem_view import read_ptr, read_int, Mem
from .minias import Bytecode

JX = 1


@dataclass
class ExtendedFrameInfo:
    """Extended frame data"""
    frame: FrameType
    o_stack_bottom: int = 0x40
    o_stack_top: int = 0x48
    o_bstack_bottom: int = 0x78
    o_bstack_size: int = 0x70

    @property
    def fid(self):
        return id(self.frame)

    @property
    def a_stack_bottom(self):
        return self.fid + self.o_stack_bottom

    @property
    def a_stack_top(self):
        return self.fid + self.o_stack_top

    @property
    def a_bstack_bottom(self):
        return self.fid + self.o_bstack_bottom

    @property
    def a_bstack_size(self):
        return self.fid + self.o_bstack_size

    def ptr_frame_stack_bottom(self):
        return read_ptr(self.a_stack_bottom)

    def ptr_frame_stack_top(self):
        return read_ptr(self.a_stack_top)

    def ptr_frame_block_stack_bottom(self):
        return self.a_bstack_bottom

    def get_bs_size(self):
        return read_int(self.a_bstack_size)

    def ptr_frame_block_stack_top(self, item_size=12):
        return self.ptr_frame_block_stack_bottom() + item_size * self.get_bs_size()


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

