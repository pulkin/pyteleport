# cython: language_level=3
from .primitives import NULL as NULL_object, block_stack_item
cdef extern from "frameobject.h":
    ctypedef struct PyTryBlock:
        int b_type
        int b_handler
        int b_level

    struct _frame:
        int f_stackdepth
        void** f_valuestack
        int f_iblock
        PyTryBlock* f_blockstack


NOTSET = object()


def get_value_stack(object frame, int depth=-1, object null=NULL_object, object until=NOTSET):
    cdef _frame* cframe = <_frame*> frame
    cdef int i
    cdef void* stack_item

    if depth == -1:
        depth = cframe.f_stackdepth  # only set for generators
    if depth == -1:
        depth = frame.f_code.co_stacksize  # max stack size

    result = []
    for i in range(depth):
        stack_item = cframe.f_valuestack[i]
        if stack_item:
            if until is <object>stack_item:
                break
            result.append(<object>stack_item)
        else:
            if until is null:
                break
            result.append(null)
    else:
        if until is not NOTSET:
            raise RuntimeError("beacon object not found")
    return result


def get_block_stack(object frame):
    cdef _frame* cframe = <_frame*> frame
    cdef int i
    cdef PyTryBlock ptb

    result = []
    for i in range(cframe.f_iblock):
        ptb = cframe.f_blockstack[i]
        result.append(block_stack_item(ptb.b_type, ptb.b_handler, ptb.b_level))
    return result
