from .mem import _unsafe_write_bytes


def test_mem_view():
    x = b"xyz"
    _unsafe_write_bytes(x, 0, b"abc")
    assert x == b"abc"
