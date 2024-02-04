import inspect
import pytest
import sys

from ..frame import FrameWrapper
from ..bytecode.opcodes import python_feature_block_stack


python_version = (sys.version_info.major, sys.version_info.minor)


def test_basic():
    frame = inspect.currentframe()
    wrapper = FrameWrapper(frame)
    assert wrapper.get_value_stack(0) == []
    locals_ = wrapper.get_locals()
    cells = wrapper.get_cells()
    for obj in frame, wrapper:
        assert obj in locals_
    assert cells == []


def test_basic_block():
    if not python_feature_block_stack:
        pytest.skip("not available for 3.11 and above")
    frame = inspect.currentframe()
    wrapper = FrameWrapper(frame)
    assert wrapper.get_block_stack() == []


def test_value_stack():
    frame = inspect.currentframe()
    wrapper = FrameWrapper(frame)
    for _ in range(1):
        for __ in [2]:
            range_iterator, list_iterator = wrapper.get_value_stack(2)
            assert "range_iterator" in str(range_iterator)
            assert "tuple_iterator" in str(list_iterator)


def test_exc_block():
    if not python_feature_block_stack:
        pytest.skip("not available for 3.11 above")
    frame = inspect.currentframe()
    wrapper = FrameWrapper(frame)

    class DummyExc(Exception):
        pass

    try:
        try:
            block_stack = wrapper.get_block_stack()
        except DummyExc:
            pass
    finally:
        pass
    assert len(block_stack) == 2


def test_generator_basic():
    def generator():
        for i in range(3):
            yield i

    gen = generator()
    next(gen)
    wrapper = FrameWrapper(gen.gi_frame)
    range_itr, = wrapper.get_value_stack()
    assert "range_iterator" in str(range_itr)
    locals_ = wrapper.get_locals()
    cells = wrapper.get_cells()
    assert locals_ == [0]
    assert cells == []
