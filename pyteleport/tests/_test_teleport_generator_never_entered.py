"""
[True] hello world
[True] vstack []
[True] bstack []
[True] 0
[True] vstack []
[True] bstack []
[True] 1
[{dry_run}] vstack []
[{dry_run}] bstack []
[{dry_run}] 0
[{dry_run}] vstack []
[{dry_run}] bstack []
[{dry_run}] 1
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_, get_tp_args, print_stack_here


setup_verbose_logging()


def generator_fn():
    print_stack_here(print_)
    yield 0
    print_stack_here(print_)
    yield 1


print_("hello world")

generator = generator_fn()
print_(next(generator))
print_(next(generator))

generator = generator_fn()
tp_dummy(**get_tp_args())
print_(next(generator))
print_(next(generator))
