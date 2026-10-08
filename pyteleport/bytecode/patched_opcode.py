from opcode import *
from .opcodes import LOAD_DEREF

# TODO: remove this file if this gets ever fixed
# cpython issue 151321: LOAD_DEREF is missing
if LOAD_DEREF not in hasfree:
    hasfree = [*hasfree, LOAD_DEREF]
if LOAD_DEREF in haslocal:
    haslocal = [i for i in haslocal if i != LOAD_DEREF]

try:
    from opcode import _inline_cache_entries
except ImportError:
    pass