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
if dump(bio):
    print_('slave')
else:
    print_('master')
    bio.seek(0)
    fun = load(bio)
    fun()
