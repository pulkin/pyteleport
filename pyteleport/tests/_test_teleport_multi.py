"""
[True] hello
[False] world 2
[False] world 1
[False] world 0
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_, get_tp_args
from time import sleep


setup_verbose_logging()
print_("hello")
fid = tp_dummy(**get_tp_args(), n=3)
sleep((2 - fid) / 10)
print_("world", fid)
