"""
[True] hello
[True] vstack []
[True] bstack []
[{dry_run}] vstack []
[{dry_run}] bstack []
[{dry_run}] world
"""
from pyteleport import tp_dummy
from pyteleport.storage import LocalStorage
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
tp_dummy(**get_tp_args(), storage=LocalStorage(loads=loads, dumps=dumps))
print_stack_here(print_)
print_("world")
