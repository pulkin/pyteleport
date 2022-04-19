"""
[True] hello
[False] world
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_, get_tp_args


setup_verbose_logging()
print_("hello")
some_object = object()
assert (some_object, tp_dummy(**get_tp_args()))[0] is some_object
print_("world")
