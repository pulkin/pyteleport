"""
[True] hello
[{dry_run}] foo
[{dry_run}] world
"""
from pyteleport import tp_dummy
from tests.helpers import setup_verbose_logging, print_stack_here, print_, get_tp_args
import pathlib


setup_verbose_logging()
print_("hello")
pathlib.secret = "foo"
tp_dummy(**get_tp_args())
print_(pathlib.secret)
print_("world")
