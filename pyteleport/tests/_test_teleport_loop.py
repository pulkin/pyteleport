"""
[True] 0
[True] 1
[True] vstack [!<class 'range_iterator'>]
[True] bstack []
[False] vstack [!<class 'range_iterator'>]
[False] bstack []
[False] 2
[False] 3
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_stack_here, print_, get_tp_args


setup_verbose_logging()
for i in range(4):
    print_(i)
    if i == 1:
        print_stack_here(print_)
        tp_dummy(**get_tp_args())
        print_stack_here(print_)
