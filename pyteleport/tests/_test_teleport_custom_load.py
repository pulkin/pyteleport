"""
[True] hello
[True] vstack []
[True] bstack []
[{dry_run}] vstack []
[{dry_run}] bstack []
[{dry_run}] world
"""
from pyteleport import tp_dummy
from pyteleport.storage import storage_protocol
from pyteleport.tests.helpers import setup_verbose_logging, print_stack_here, print_, get_tp_args
import dill
import gzip


def dumps(obj):
    return gzip.compress(dill.dumps(obj))


def loads(data):
    from dill import loads
    from gzip import decompress
    return loads(decompress(data))


setup_verbose_logging()
print_("hello")
print_stack_here(print_)
tp_dummy(**get_tp_args(), object_storage_protocol=storage_protocol(load_from_code=loads, save_to_code=dumps,
                                                                   load_on_startup=None))
print_stack_here(print_)
print_("world")
