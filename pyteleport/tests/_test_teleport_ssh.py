"""
[True] hello
[True] vstack []
[True] bstack []
[{dry_run}] vstack []
[{dry_run}] bstack []
[{dry_run}] world
"""
from pyteleport import tp_shell
from pyteleport.storage import socket_transmission_engine
from pyteleport.tests.helpers import setup_verbose_logging, print_stack_here, print_, get_tp_args


setup_verbose_logging()
print_("hello")
print_stack_here(print_)
tp_shell("ssh", "-o BatchMode=yes", "-o StrictHostKeyChecking=no", "-o UserKnownHostsFile=/dev/null",
         "-R {port}:localhost:{port}", f"ubuntu@ec2-184-73-122-159.compute-1.amazonaws.com", python="python3",
         object_storage_protocol=socket_transmission_engine, **get_tp_args())
print_stack_here(print_)
print_("world")
