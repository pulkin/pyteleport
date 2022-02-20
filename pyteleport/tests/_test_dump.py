"""
[True] hello
[True] master
[True] slave
"""
from pyteleport.tests.helpers import setup_verbose_logging, print_
from pyteleport.snapshot import dump, load

from io import BytesIO


setup_verbose_logging()
print_('hello')
bio = BytesIO()
dump(bio)

if "__pyteleport_stack_level__" in locals():
    print_('slave')
else:
    print_('master')
    bio.seek(0)
    fun = load(bio)
    fun()
