import mem_view


def test_mem_view():
    x = b"xyz"
    v = mem_view.Mem.view(x)
    v[:] = b"abc"
    assert x == "abc".encode()

