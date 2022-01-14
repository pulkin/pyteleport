"""
[True] vstack []
[True] bstack []
[True] hello 0
[False] world 1
[False] vstack []
[False] bstack []
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_, get_tp_args, print_stack_here


setup_verbose_logging()


def generator_fn_inner():
    yield 0
    yield 1


def generator_fn():
    print_stack_here(print_)
    yield from generator_fn_inner()
    print_stack_here(print_)
    yield "yield"


generator = generator_fn()
print_("hello", next(generator))
tp_dummy(**get_tp_args())
print_("world", next(generator))
assert next(generator) == "yield"
