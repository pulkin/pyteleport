# cython: language_level=3
from cpython.version cimport PY_VERSION_HEX
from cpython.bytes cimport PyBytes_AsString, PyBytes_FromStringAndSize
from cpython.ref cimport PyObject
from .primitives import NULL as NULL_object, block_stack_item


cdef extern from "frameobject.h":
    ctypedef struct PyTryBlock:
        int b_type
        int b_handler
        int b_level

    struct _frame:
        PyObject** f_valuestack
        PyObject** f_stacktop  # available in 3.9 and earlier
        int f_stackdepth  # available in 3.10 and later
        PyTryBlock* f_blockstack
        int f_iblock


cdef extern from *:  # stack depth for different python versions
    """
    #if PY_VERSION_HEX >= 0x030A0000
      static int _pyteleport_stackdepth(struct _frame* frame) {return frame->f_stackdepth;}
    #elif PY_VERSION_HEX >= 0x03080000
      static int _pyteleport_stackdepth(struct _frame* frame) {
        if (frame->f_stacktop)
          return (int) (frame->f_stacktop - frame->f_valuestack);
        else
          return -1;
      }
    #elif defined(PY_VERSION_HEX)
      #error "Unknown python version"
    #else
      #error "PY_VERSION_HEX not defined"
    #endif
    """
    int _pyteleport_stackdepth(_frame* frame)


NOTSET = object()


def snapshot_value_stack(object frame):
    cdef _frame* cframe = <_frame*> frame
    cdef int i

    cdef int depth = frame.f_code.co_stacksize  # max stack size
    return PyBytes_FromStringAndSize(<char*>cframe.f_valuestack, sizeof(void*) * depth)


def get_value_stack(
        object frame,
        int depth=-1,
        object null=NULL_object,
        object until=NOTSET,
        object valuestack=None,
):
    cdef _frame* cframe = <_frame*> frame
    cdef int i
    cdef PyObject** f_valuestack
    cdef PyObject* stack_item
    cdef int auto_depth_offset = 0

    if depth < 0:
        auto_depth_offset = -1 - depth

    if valuestack is not None:
        f_valuestack = <PyObject**>PyBytes_AsString(valuestack)
    else:
        f_valuestack = cframe.f_valuestack

    if depth < 0:
        depth = _pyteleport_stackdepth(cframe)  # only works for inactive generator frames
    if depth < 0:
        depth = frame.f_code.co_stacksize  # max stack size
    depth += auto_depth_offset

    result = []
    for i in range(depth):
        stack_item = f_valuestack[i]
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
