"""
[True] hello
[True] g: in_f_after
[{dry_run}] g: in_f_after
[{dry_run}] g: in_g
[{dry_run}] f: in_g
[{dry_run}] main: in_main
[{dry_run}] world
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_, get_tp_args


setup_verbose_logging()
print_("hello")

o = "in_main"


def f():
    o = "in_f"

    def g():
        nonlocal o
        print_(f"g: {o}")
        tp_dummy(**get_tp_args())
        print_(f"g: {o}")
        o = "in_g"
        print_(f"g: {o}")

    o = "in_f_after"
    g()
    print_(f"f: {o}")


f()
print_(f"main: {o}")
print_("world")
