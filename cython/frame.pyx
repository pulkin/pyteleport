# cython: language_level=3
from cpython.version cimport PY_VERSION_HEX
from cpython.ref cimport PyObject
from .primitives import NULL as NULL_object, block_stack_item


cdef extern from "frameobject.h":
    ctypedef struct PyFrameObject:
        PyObject** f_valuestack
        PyObject** f_localsplus


cdef extern from *:
    """
    #ifndef PY_VERSION_HEX
        #error "PY_VERSION_HEX not defined"
    #else
        #define PYTELEPORT_PYTHON_VERSION (PY_VERSION_HEX >> 16)
    #endif

    #if PYTELEPORT_PYTHON_VERSION == 0x030B
        #include "internal/pycore_frame.h"

        #define FRAME (frame->f_frame)
        #define CODE (FRAME->f_code)
        static PyObject** _pyframe_get_value_stack(PyFrameObject* frame) {return FRAME->localsplus + CODE->co_nlocalsplus;}
        static int _pyframe_get_value_stack_depth(PyFrameObject* frame) {return FRAME->stacktop;}

        #define _PYFRAME_DEFINE_BLOCK_STACK_GETTER(name) static int _pyframe_get_block_stack_ ## name(PyFrameObject* frame, int i) {return -1;}
        static int _pyframe_get_block_stack_depth(PyFrameObject* frame) {return -1;}

        static int _pyframe_n_locals(PyFrameObject* frame) {return CODE->co_nlocals;}
        static int _pyframe_n_cells(PyFrameObject* frame) {return CODE->co_nlocalsplus - CODE->co_nlocals;}

        static PyObject** _pyframe_get_locals(PyFrameObject* frame) {return FRAME->localsplus;}
        static PyObject** _pyframe_get_cells(PyFrameObject* frame) {return FRAME->localsplus + CODE->co_nlocals;}

    #elif PYTELEPORT_PYTHON_VERSION == 0x030A
        #include "tupleobject.h"

        static PyObject** _pyframe_get_value_stack(PyFrameObject* frame) {return frame->f_valuestack;}
        static int _pyframe_get_value_stack_depth(PyFrameObject* frame) {return frame->f_stackdepth;}

        #define _PYFRAME_DEFINE_BLOCK_STACK_GETTER(name) static int _pyframe_get_block_stack_ ## name(PyFrameObject* frame, int i) {return frame->f_blockstack[i].name;}
        static int _pyframe_get_block_stack_depth(PyFrameObject* frame) {return frame->f_iblock;}

        static PyObject** _pyframe_get_locals(PyFrameObject* frame) {return frame->f_localsplus;}
        static PyObject** _pyframe_get_cells(PyFrameObject* frame) {return frame->f_localsplus + frame->f_code->co_nlocals;}

        static int _pyframe_n_locals(PyFrameObject* frame) {return frame->f_code->co_nlocals;}
        static int _pyframe_n_cells(PyFrameObject* frame) {return PyTuple_Size(frame->f_code->co_freevars) + PyTuple_Size(frame->f_code->co_cellvars);}

    #elif PYTELEPORT_PYTHON_VERSION == 0x0309

        static PyObject** _pyframe_get_value_stack(PyFrameObject* frame) {return frame->f_valuestack;}
        static int _pyframe_get_value_stack_depth(PyFrameObject* frame) {
            if (frame->f_stacktop)
                return (int) (frame->f_stacktop - frame->f_valuestack);
            else
                return -1;
        }

        #define _PYFRAME_DEFINE_BLOCK_STACK_GETTER(name) static int _pyframe_get_block_stack_ ## name(PyFrameObject* frame, int i) {return frame->f_blockstack[i].name;}
        static int _pyframe_get_block_stack_depth(PyFrameObject* frame) {return frame->f_iblock;}

        static PyObject** _pyframe_get_locals(PyFrameObject* frame) {return frame->f_localsplus;}
        static PyObject** _pyframe_get_cells(PyFrameObject* frame) {return frame->f_localsplus + frame->f_code->co_nlocals;}

        static int _pyframe_n_locals(PyFrameObject* frame) {return frame->f_code->co_nlocals;}
        static int _pyframe_n_cells(PyFrameObject* frame) {return PyTuple_Size(frame->f_code->co_freevars) + PyTuple_Size(frame->f_code->co_cellvars);}

    #else
        #error "Not implemented for this cpython version"
    #endif

    _PYFRAME_DEFINE_BLOCK_STACK_GETTER(b_type)
    _PYFRAME_DEFINE_BLOCK_STACK_GETTER(b_handler)
    _PYFRAME_DEFINE_BLOCK_STACK_GETTER(b_level)
    """

    PyObject** _pyframe_get_value_stack(PyFrameObject* frame)
    int _pyframe_get_value_stack_depth(PyFrameObject* frame)
    int _pyframe_get_block_stack_b_type(PyFrameObject* frame, int i)
    int _pyframe_get_block_stack_b_handler(PyFrameObject* frame, int i)
    int _pyframe_get_block_stack_b_level(PyFrameObject* frame, int i)
    int _pyframe_get_block_stack_depth(PyFrameObject* frame)

    PyObject** _pyframe_get_locals(PyFrameObject* frame)
    PyObject** _pyframe_get_cells(PyFrameObject* frame)
    int _pyframe_n_locals(PyFrameObject* frame)
    int _pyframe_n_cells(PyFrameObject* frame)


NOTSET = object()


cdef class FrameWrapper:
    cdef PyFrameObject* frame

    def __cinit__(self, object frame):
        self.frame = <PyFrameObject*>frame

    @property
    def block_stack(self):
        cdef:
            int i
        if _pyframe_get_block_stack_depth(self.frame) == -1:
            raise ValueError("not implemented for this python version")

        result = []
        for i in range(_pyframe_get_block_stack_depth(self.frame)):
            result.append(block_stack_item(
                _pyframe_get_block_stack_b_type(self.frame, i),
                _pyframe_get_block_stack_b_handler(self.frame, i),
                _pyframe_get_block_stack_b_level(self.frame, i),
            ))
        return result

    def get_value_stack(self, int stack_size = -1, object null = NULL_object):
        cdef:
            int i
            PyObject* stack_item

        # first, determine the stack size
        if stack_size < 0:
            stack_size = _pyframe_get_value_stack_depth(self.frame)  # only works for inactive generator frames
            if stack_size < 0:
                raise ValueError("this frame requires stack size")

        # second, copy stack objects
        result = []
        for i in range(stack_size):
            stack_item = _pyframe_get_value_stack(self.frame)[i]
            if stack_item:
                result.append(<object>stack_item)
            else:
                result.append(null)
        return result

    def get_locals(self, object null = NULL_object):
        cdef:
            PyObject** ptr = _pyframe_get_locals(self.frame)
            int i

        result = []
        for i in range(_pyframe_n_locals(self.frame)):
            if ptr[i]:
                result.append(<object>ptr[i])
            else:
                result.append(null)
        return result

    def get_cells(self, object null = NULL_object):
        cdef:
            PyObject** ptr = _pyframe_get_cells(self.frame)
            int i

        result = []
        for i in range(_pyframe_n_cells(self.frame)):
            if ptr[i]:
                result.append(<object>ptr[i])
            else:
                result.append(null)
        return result
