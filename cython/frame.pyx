# cython: language_level=3
from cpython.version cimport PY_VERSION_HEX
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
        int f_stackdepth  # available in 3.10
        PyTryBlock* f_blockstack
        int f_iblock
        PyObject** f_localsplus


cdef extern from *:  # stack depth for different python versions
    """
    #if PY_VERSION_HEX >= 0x030A0000
      static int _pyteleport_stackdepth(struct _frame* frame) {
        return frame->f_stackdepth;
      }
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


cdef class FrameWrapper:
    cdef _frame* frame

    def __cinit__(self, object frame):
        self.frame = <_frame*>frame

    @property
    def block_stack(self):
        cdef:
            int i
            PyTryBlock ptb

        result = []
        for i in range(self.frame.f_iblock):
            ptb = self.frame.f_blockstack[i]
            result.append(block_stack_item(ptb.b_type, ptb.b_handler, ptb.b_level))
        return result

    def get_value_stack(self, int stack_size = -1, object null = NULL_object):
        cdef:
            int i
            PyObject* stack_item

        # first, determine the stack size
        if stack_size < 0:
            stack_size = _pyteleport_stackdepth(self.frame)  # only works for inactive generator frames
            if stack_size < 0:
                raise ValueError("this frame requires stack size")

        # second, copy stack objects
        result = []
        for i in range(stack_size):
            stack_item = self.frame.f_valuestack[i]
            if stack_item:
                result.append(<object>stack_item)
            else:
                result.append(null)
        return result

    def get_locals_plus(self, object null=NULL_object):
        cdef:
            PyObject* item
            int i, n_locals

        code = (<object>self.frame).f_code
        n_locals = code.co_nlocals
        assert len(code.co_varnames) == n_locals
        cdef:
            int n_cells = len(code.co_cellvars)
            int n_free = len(code.co_freevars)

        locals = []
        for i in range(n_locals + n_cells + n_free):
            item = self.frame.f_localsplus[i]
            if item:
                locals.append(<object>item)
            else:
                locals.append(null)

        return locals[:n_locals], locals[n_locals:n_locals + n_cells], locals[n_locals + n_cells:]
