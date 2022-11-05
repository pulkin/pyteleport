"""
[True] hello
[{dry_run}] f: test_o is o=False
[{dry_run}] f: test_o is o_f=False
[{dry_run}] f: o is o_f=True
[{dry_run}] __main__: test_o is o=True
[{dry_run}] __main__: test_o is o_global=False
[{dry_run}] __main__: o is o_global=False
[{dry_run}] world
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_, get_tp_args


setup_verbose_logging()
print_("hello")

o = o_global = object()


def f():
    o = o_f = object()

    def g():
        global o
        o = object()
        return o, tp_dummy(**get_tp_args())

    test_o, _ = g()
    print_(f"f: {test_o is o=}")
    print_(f"f: {test_o is o_f=}")
    print_(f"f: {o is o_f=}")
    return test_o


test_o = f()
print_(f"__main__: {test_o is o=}")
print_(f"__main__: {test_o is o_global=}")
print_(f"__main__: {o is o_global=}")
print_("world")
