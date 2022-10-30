"""
[True] try
[True] teleport
[True] vstack []
[True] bstack [122/0]
[{dry_run}] vstack []
[{dry_run}] bstack [122/0]
[{dry_run}] raise
[{dry_run}] CustomException('hello')
[{dry_run}] handle
[{dry_run}] done
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_stack_here, print_, get_tp_args


setup_verbose_logging()


class CustomException(Exception):
    pass


print_("try")
try:
    print_("teleport")
    print_stack_here(print_)
    tp_dummy(**get_tp_args())
    print_stack_here(print_)
    print_("raise")
    raise CustomException("hello")
    print_("unreachable")
except CustomException as e:
    print_(repr(e))
    print_("handle")
print_("done")
