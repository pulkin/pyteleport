"""
[True] hello
[True] None
[True] 1
[True] 2
[True] 3
[{dry_run}] 3
[{dry_run}] 1
[{dry_run}] 2
[{dry_run}] 3
[{dry_run}] world
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_, get_tp_args


setup_verbose_logging()
print_("hello")


def prepare():
    o = None

    def get():
        nonlocal o
        return o

    def set_one():
        nonlocal o
        o = 1

    def set_two():
        nonlocal o
        o = 2

    def _compute():
        def set_3():
            nonlocal o
            o = 3
        return set_3

    return get, set_one, set_two, _compute()


get, set_one, set_two, set_three = prepare()

print_(get())
set_one()
print_(get())
set_two()
print_(get())
set_three()
print_(get())

tp_dummy(**get_tp_args())

print_(get())
set_one()
print_(get())
set_two()
print_(get())
set_three()
print_(get())
print_("world")
