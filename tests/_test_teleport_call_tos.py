"""
[True] hello
[True] vstack []
[True] bstack {empty_bstack}
[{dry_run}] {None if dry_run else 42}
[{dry_run}] vstack []
[{dry_run}] bstack {empty_bstack}
[{dry_run}] world
"""
from pyteleport import tp_dummy
from tests.helpers import setup_verbose_logging, print_stack_here, print_, get_tp_args


setup_verbose_logging()
print_("hello")
print_stack_here(print_)
print_((tp_dummy(**get_tp_args(), return_=[lambda: 42]) or (lambda: None))())
print_stack_here(print_)
print_("world")
