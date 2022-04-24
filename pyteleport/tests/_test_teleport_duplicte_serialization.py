"""
[True] hello
[False] world
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_, get_tp_args


some_object = object()


def f():
    a = some_object

    def g():
        return some_object, tp_dummy(**get_tp_args())

    b, _ = g()
    assert b is a
    assert b is some_object
    assert a is some_object


setup_verbose_logging()
print_("hello")
f()
print_("world")
