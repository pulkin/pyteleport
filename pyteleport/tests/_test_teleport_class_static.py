"""
[True] hello
[True] vstack [NULL, !<class 'function'>]
[True] bstack []
[False] vstack [NULL, !<class 'function'>, None, 3, 4]
[False] bstack []
[False] world
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_stack_here, print_, get_tp_args


class SomeClass:
    @staticmethod
    def sum(_1, x, y, _2):
        return x + y


def f():
    tp_dummy(**get_tp_args())
    return 3


setup_verbose_logging()
instance = SomeClass()
print_("hello")
assert instance.sum(print_stack_here(print_), f(), 4, print_stack_here(print_)) == 7
print_("world")
