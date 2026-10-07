"""
[True] with
[True] <TestContext> enter
[True] teleport
{"[True] vstack [!<class 'function'>, !<class '__main__.TestContext'>]" if py >= 0x030E else "[True] vstack [!<class 'method'>]"}
[True] bstack {'[122/1]' if py < 0x30B else '--'}
{f"[{dry_run}] vstack [!<class 'function'>, !<class '__main__.TestContext'>]" if py >= 0x030E else f"[{dry_run}] vstack [!<class 'method'>]"}
[{dry_run}] bstack {'[122/1]' if py < 0x30B else '--'}
[{dry_run}] <TestContext> exit
[{dry_run}] done
"""
from pyteleport import tp_dummy
from tests.helpers import setup_verbose_logging, print_stack_here, print_, get_tp_args


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
