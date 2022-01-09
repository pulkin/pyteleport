"""
[True] with
[True] <TestContext> enter
[True] teleport
[True] vstack [!<class 'method'>]
[True] bstack [122/1]
[False] vstack [!<class 'method'>]
[False] bstack [122/1]
[False] <TestContext> exit
[False] done
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_stack_here, print_, get_tp_args


setup_verbose_logging()


class CustomException(Exception):
    pass


class TestContext:
    def __enter__(self):
        print_("<TestContext> enter")

    def __exit__(self, *args):
        print_("<TestContext> exit")


print_("with")
with TestContext():
    print_("teleport")
    print_stack_here(print_)
    tp_dummy(**get_tp_args())
    print_stack_here(print_)
print_("done")
