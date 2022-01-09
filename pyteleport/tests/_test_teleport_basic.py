"""
[True] hello
[True] vstack []
[True] bstack []
[False] vstack []
[False] bstack []
[False] world
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_stack_here, print_, get_tp_args


setup_verbose_logging()
print_("hello")
print_stack_here(print_)
tp_dummy(**get_tp_args())
print_stack_here(print_)
print_("world")
