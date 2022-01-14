"""
[True] vstack [!<class 'range_iterator'>]
[True] bstack []
[True] hello 0
[False] vstack [!<class 'range_iterator'>]
[False] bstack []
[False] world 1
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_, get_tp_args, print_stack_here


setup_verbose_logging()


def generator_fn():
    for i in range(2):
        print_stack_here(print_)
        yield i


generator = generator_fn()
print_("hello", next(generator))
tp_dummy(**get_tp_args())
print_("world", next(generator))
