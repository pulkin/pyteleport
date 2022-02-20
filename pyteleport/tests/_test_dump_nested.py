"""
[True] hello
[True] entered
[True] exited
[True] OK
"""
from pyteleport.tests.helpers import setup_verbose_logging, print_
from pyteleport.snapshot import dump, load

from io import BytesIO
from sys import exit


setup_verbose_logging()
print_('hello')
bio = BytesIO()


def a():
    def b():
        def c():
            print_("entered")
            result = "hello"
            r2 = "world"
            if dump(bio) is None:
                bio.seek(0)
                fun = load(bio)
                fun()
                exit()
            assert result == "hello"
            assert r2 == "world"
            print_("exited")
            return result + " world"
        return len(c()) + float("3.5")
    return 5 * (3 + b())
assert a() == 87.5
print_("OK")
