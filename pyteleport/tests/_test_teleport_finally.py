"""
[True] try
[True] raise
[True] handle
[True] teleport
[True] vstack [{'NULL' if python_version <= 0x0308 else ''}]
[True] bstack []
[False] vstack [{'NULL' if python_version <= 0x0308 else ''}]
[False] bstack []
[False] finally
[False] done
"""
from pyteleport import tp_dummy
from pyteleport.tests import helpers  # TODO: the module needs to be pickled in order to save pid_on_init
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
    print_("handle")
finally:
    print_("teleport")
    print_stack_here()
    tp_dummy(**get_tp_args())
    print_stack_here()
    print_("finally")
print_("done")
