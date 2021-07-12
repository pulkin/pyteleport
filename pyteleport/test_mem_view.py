from .mem_view import Mem


def test_mem_view():
    x = b"xyz"
    v = Mem.view(x)
    v[:] = b"abc"
    assert x == "abc".encode()

