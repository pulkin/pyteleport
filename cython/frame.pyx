# cython: language_level=3
from .primitives import NULL as NULL_object
cdef extern from "frameobject.h":
    struct _frame:
        int f_stackdepth
        void** f_valuestack


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
