"""
[True] try
[True] raise
[True] vstack [NULL, NULL, None]
[True] bstack [122/0, 257/0, 122/3]
[True] teleport
[{dry_run}] vstack [NULL, NULL, None]
[{dry_run}] bstack [122/0, 257/0, 122/3]
[{dry_run}] handle
[{dry_run}] finally
[{dry_run}] done
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_stack_here, print_, get_tp_args


setup_verbose_logging()


class CustomException(Exception):
    pass


print_("try")
try:
    print_("raise")
    raise CustomException("hello")
    print_("unreachable")
except CustomException as e:
    print_stack_here(print_)
    print_("teleport")
    tp_dummy(**get_tp_args())
    print_stack_here(print_)
    print_("handle")
finally:
    print_("finally")
print_("done")
