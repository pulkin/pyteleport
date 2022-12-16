"""
[True] hello
[True] vstack []
[True] bstack []
[{dry_run}] vstack []
[{dry_run}] bstack []
[{dry_run}] world
"""
from pyteleport.experimental import tp_disposable_ec2
from pyteleport.tests.helpers import setup_verbose_logging, print_stack_here, print_, get_tp_args
from pyteleport.storage import socket_transmission_engine
import dill
from ssl import SSLContext, SSLSocket
for t in SSLContext, SSLSocket:
    dill.register(t)(lambda pickler, obj: pickler.save_reduce(lambda: None, tuple(), obj=obj))


setup_verbose_logging()
print_("hello")
print_stack_here(print_)
tp_disposable_ec2(allocate_kwargs={"LaunchTemplateId": "lt-02d2bc621b78d5b8b"}, **get_tp_args(),
                  ec2_username="ubuntu", python="python3",
                  object_storage_protocol=socket_transmission_engine)
print_stack_here(print_)
print_("world")
