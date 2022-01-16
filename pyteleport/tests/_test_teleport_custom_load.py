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
tp_dummy(**get_tp_args(), pack_object=dumps, unpack_object=loads)
print_stack_here(print_)
print_("world")
