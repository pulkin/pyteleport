"""
[True] hello
[True] vstack []
[True] bstack []
[{dry_run}] vstack []
[{dry_run}] bstack []
[{dry_run}] world
"""
from pyteleport import tp_dummy
from pyteleport.storage import socket_transmission_engine
from pyteleport.tests.helpers import setup_verbose_logging, print_stack_here, print_, get_tp_args


setup_verbose_logging()
print_("hello")
print_stack_here(print_)
tp_dummy(**get_tp_args(), object_storage_protocol=socket_transmission_engine)
print_stack_here(print_)
print_("world")
