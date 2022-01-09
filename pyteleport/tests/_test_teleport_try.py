"""
[True] try
[True] teleport
[True] vstack []
[True] bstack [122/0]
[False] vstack []
[False] bstack [122/0]
[False] raise
[False] CustomException('hello')
[False] handle
[False] done
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
