"""
[True] try
[True] raise
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
    try:
        print_("raise")
        raise CustomException("hello")
        print_("unreachable")
    except AnotherException:
        pass
except CustomException as e:
    print_stack_here(print_)
    print_("teleport")
    tp_dummy(**get_tp_args())
    print_stack_here(print_)
    print_("handle")
finally:
    print_("finally")
print_("done")
