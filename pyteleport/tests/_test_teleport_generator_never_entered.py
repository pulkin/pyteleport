"""
[{dry_run}] hello world 0 1
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_, get_tp_args, print_stack_here


setup_verbose_logging()


def generator_fn():
    yield 0
    yield 1


generator = generator_fn()
tp_dummy(**get_tp_args())
print_("hello world", next(generator), next(generator))
