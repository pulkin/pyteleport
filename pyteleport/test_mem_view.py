from .mem import _unsafe_write_bytes


def test_mem_view():
    x = b"xyz"
    _unsafe_write_bytes(x, b"abc", 0)
    assert x == b"abc"
