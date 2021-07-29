from ctypes import memmove, string_at
import struct


def ptr_bytes_data(data):
    """Points to the contents of bytes built-in"""
    return id(data) + 0x20


def ptr_frame_stack_bottom(data):
    """Points to the bottom of frame stack"""
    # https://github.com/python/cpython/blob/46b16d0bdbb1722daed10389e27226a2370f1635/Include/cpython/frameobject.h#L17
    result_star = id(data) + 0x40
    result, = struct.unpack("P", Mem(result_star, 0x08)[:])
    return result


def ptr_frame_stack_top(data):
    """Points after the top of frame stack"""
    # https://github.com/python/cpython/blob/46b16d0bdbb1722daed10389e27226a2370f1635/Include/cpython/frameobject.h#L17
    result_star = id(data) + 0x48
    result, = struct.unpack("P", Mem(result_star, 0x08)[:])
    return result


def _p_hex(x):
    return '\n'.join(f"+{offset:04x}: {' '.join(f'{i:02x}' for i in x[offset:offset + 16])}" for offset in range(0, len(x), 16))


class Mem:
    def __init__(self, addr, length):
        self.addr = addr
        self.length = length

    @property
    def _bytes(self):
        return string_at(self.addr, self.length)

    def _w(self, offset, buffer):
        memmove(self.addr + offset, ptr_bytes_data(buffer), len(buffer))

    def __getitem__(self, item):
        return self._bytes[item]

    def __setitem__(self, item, value):
        if isinstance(value, int):
            value = bytes([value & 0xFF])
        else:
            value = bytes(value)
        if isinstance(item, int):
            assert len(value) == 1
            self._w(item, value)
        elif isinstance(item, slice):
            start, stop, step = item.indices(self.length)
            assert step == 1
            assert len(value) == stop - start
            self._w(start, value)
        else:
            raise NotImplementedError

    def __len__(self):
        return self.length

    def __repr__(self):
        return f"Mem({self._bytes})"

    def __str__(self):
        return f"Mem @0x{self.addr:016x}\n" + _p_hex(self)

    @staticmethod
    def view(a):
        if isinstance(a, bytes):
            return Mem(ptr_bytes_data(a), len(a))
        else:
            raise NotImplementedError

