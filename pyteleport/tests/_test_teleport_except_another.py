"""
[True] try
[True] inner try
[True] raise
[True] raise when handling
[True] vstack [NULL, NULL, None]
[True] bstack [122/0, 257/0, 122/3]
[True] teleport
[False] vstack [NULL, NULL, None]
[False] bstack [122/0, 257/0, 122/3]
[False] handle
[False] finally
[False] done
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_stack_here, print_, get_tp_args


setup_verbose_logging()


class CustomException(Exception):
    pass


class AnotherException(Exception):
    pass


print_("try")
try:
    print_("inner try")
    try:
        print_("raise")
        raise CustomException("hello")
        print_("unreachable")
    except CustomException as e:
        print_("raise when handling")
        raise AnotherException("world")
        print_("unreachable")
except AnotherException as e:
    print_stack_here(print_)
    print_("teleport")
    tp_dummy(**get_tp_args())
    print_stack_here(print_)
    print_("handle")
finally:
    print_("finally")
print_("done")
