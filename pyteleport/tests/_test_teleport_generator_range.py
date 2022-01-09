"""
[True] hello 0
[False] world 1
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_, get_tp_args


setup_verbose_logging()
generator = iter(range(2))

print_("hello", next(generator))
tp_dummy(**get_tp_args())
print_("world", next(generator))
