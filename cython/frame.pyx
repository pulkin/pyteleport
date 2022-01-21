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
    return PyBytes_FromStringAndSize(<char*>cframe.f_valuestack, sizeof(PyObject*) * depth)


def get_value_stack_size(object frame, object until=NOTSET):
    cdef _frame* cframe = <_frame*> frame
    cdef int result, i
    cdef PyObject* stack_item

    result = _pyteleport_stackdepth(cframe)  # only works for inactive generator frames
    if result >= 0:
        return result
    result = frame.f_code.co_stacksize  # max stack size
    if until is NOTSET:
        return result
    for i in range(result):
        stack_item = cframe.f_valuestack[i]
        if until is <object>stack_item:
            return i
    raise ValueError("beacon object not found on value stack")


def get_value_stack(
        object value_stack,
        int size,
        object null=NULL_object,
):
    cdef PyObject** cvalue_stack = <PyObject**>PyBytes_AsString(value_stack)
    cdef PyObject* stack_item
    cdef int i

    result = []
    for i in range(size):
        stack_item = cvalue_stack[i]
        if stack_item:
            result.append(<object>stack_item)
        else:
            result.append(null)
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
