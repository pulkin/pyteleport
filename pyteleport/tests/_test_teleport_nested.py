"""
[True] entered
[False] exited
[False] OK
"""
from pyteleport import tp_dummy
from pyteleport.tests.helpers import setup_verbose_logging, print_, get_tp_args


setup_verbose_logging()


def a():
    def b():
        def c():
            print_("entered")
            result = "hello"
            r2 = "world"
            tp_dummy(**get_tp_args())
            assert result == "hello"
            assert r2 == "world"
            print_("exited")
            return result + " world"
        return len(c()) + float("3.5")
    return 5 * (3 + b())


assert a() == 87.5
print_("OK")
