import inspect

from pyteleport.frame import FrameWrapper


def test_basic():
    frame = inspect.currentframe()
    wrapper = FrameWrapper(frame)
    assert wrapper.block_stack == []
    assert wrapper.get_value_stack(2) == [wrapper.get_value_stack, 2]
    locals_, cells, free = wrapper.get_locals_plus()
    for obj in frame, wrapper:
        assert obj in locals_
    assert cells == []
    assert free == []


def test_generator_basic():
    def generator():
        for i in range(3):
            yield i

    gen = generator()
    next(gen)
    wrapper = FrameWrapper(gen.gi_frame)
    assert wrapper.block_stack == []
    range_itr, = wrapper.get_value_stack()
    assert "range_iterator" in str(range_itr)
    locals_, cells, free = wrapper.get_locals_plus()
    assert locals_ == [0]
    assert cells == []
    assert free == []
