# cython: language_level=3
from cpython.bytes cimport PyBytes_Size, PyBytes_AsString
from libc.string cimport memcpy


def _unsafe_write_bytes(bytes destination, bytes value, int offset):
    if offset < 0:
        raise ValueError(f"offset = {offset} < 0")
    cdef int dsize = PyBytes_Size(destination)
    cdef void* dbuf = PyBytes_AsString(destination)
    cdef int vsize = PyBytes_Size(value)
    cdef void* vbuf = PyBytes_AsString(value)
    if offset + vsize > dsize:
        raise ValueError(f"offset + len(value) = {offset + vsize} > len(destination) = {dsize}")
    memcpy(dbuf + offset, vbuf, vsize)
