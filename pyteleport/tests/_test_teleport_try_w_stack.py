"""
[True] loop 0
[True] try
[True] teleport
[True] vstack [!<class 'range_iterator'>, !<class 'range_iterator'>]
[True] bstack [122/1, 122/1]
[False] vstack [!<class 'range_iterator'>, !<class 'range_iterator'>]
[False] bstack [122/1, 122/1]
[False] raise
[False] CustomException('hello')
[False] handle
[False] finally
[False] loop 1
[False] loop 2
[False] done
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_stack_here, print_, get_tp_args


setup_verbose_logging()


class CustomException(Exception):
    pass


for j in range(3):
    print_(f"loop {j}")
    if j == 0:
        print_(f"try")
        try:
            for i in range(3, 6):
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
        finally:
            print_("finally")
print_("done")
