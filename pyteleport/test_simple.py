from subprocess import check_output
from tempfile import NamedTemporaryFile
import sys

import pytest


def run_python(script):
    return check_output([sys.executable], input=script, text=True)


@pytest.fixture()
def env_getter(pytestconfig):
    if pytestconfig.getoption("inherit_path"):
        return '\n'.join(['import os, sys', 'env = os.environ.copy()', 'env["PYTHONPATH"] = ":".join([os.getcwd(), *sys.path])'])
    else:
        return 'env = None'


def test_trivial():
    dump = NamedTemporaryFile()
    assert run_python(
f"""
from pyteleport.core import dump
print('hello')
dump(open("{dump.name}", 'wb'))
print('world')
""") == 'hello\n'
    assert run_python(
f"""
from pyteleport.core import load
load(open("{dump.name}", 'rb'))()
""") == 'world\n'


def test_nested():
    dump = NamedTemporaryFile()
    assert run_python(
f"""
from pyteleport.core import dump
def a():
    def b():
        def c():
            print("entered")
            result = "hello"
            r2 = "world"
            dump(open("{dump.name}", 'wb'))
            assert result == "hello"
            assert r2 == "world"
            print("exited")
            return result + " world"
        return len(c()) + float("3.5")
    return 5 * (3 + b())
assert a() == 87.5
print("OK")
""") == "entered\n"
    assert run_python(
f"""
from pyteleport.core import load
load(open("{dump.name}", 'rb'))()
""") == 'exited\nOK\n'


test_preamble = f"""
from pyteleport import dummy_teleport
import os

parent_pid = os.getpid()

def log(*args):
    print(f"[{{os.getpid() == parent_pid}}]", *args, flush=True)
"""

def test_simple_teleport(env_getter):
    assert run_python(
f"""
{test_preamble}
{env_getter}
log("hello")
dummy_teleport(env=env)
log("world")
""") == "[True] hello\n[False] world\n"

def test_nested_teleport(env_getter):
    assert run_python(
f"""
{test_preamble}
{env_getter}

def a():
    def b():
        def c():
            log("entered")
            result = "hello"
            r2 = "world"
            dummy_teleport(env=env)
            assert result == "hello"
            assert r2 == "world"
            log("exited")
            return result + " world"
        return len(c()) + float("3.5")
    return 5 * (3 + b())
assert a() == 87.5
log("OK")
""") == """[True] entered
[False] exited
[False] OK
"""

def test_nested_teleport_w_globals(env_getter):
    assert run_python(
f"""
{test_preamble}
{env_getter}

def a():
    def b():
        def c():
            log("entered c")
            result = "hello"
            dummy_teleport(env=env)
            log("exiting c")
            return result + " world"
        return len(c()) + float("3.5")
    return 5 * (3 + b())

log("hi")
assert a() == 87.5
log("bye")
""") == """[True] hi
[True] entered c
[False] exiting c
[False] bye
"""

def test_simple_teleport_builtin_range(env_getter):
    assert run_python(
f"""
{test_preamble}
{env_getter}
generator = iter(range(2))

log("hello", next(generator))
dummy_teleport(env=env)
log("world", next(generator))
""") == "[True] hello 0\n[False] world 1\n"

def test_simple_teleport_builtin_count(env_getter):
    assert run_python(
f"""
from itertools import count
{test_preamble}
{env_getter}
generator = iter(count())

def log(*args):
    print(f"[{{os.getpid() == parent_pid}}]", *args, flush=True)

log("hello", next(generator))
dummy_teleport(env=env)
log("world", next(generator))
""") == "[True] hello 0\n[False] world 1\n"


def test_simple_teleport_generator(env_getter):
    assert run_python(
f"""
{test_preamble}
{env_getter}

def generator_fn():
    yield 0
    yield 1
generator = generator_fn()

log("hello", next(generator))
dummy_teleport(env=env)
log("world", next(generator))
""") == "[True] hello 0\n[False] world 1\n"


def test_simple_loop(env_getter):
    assert run_python(
f"""
{test_preamble}
{env_getter}

for i in range(4):
    log(i)
    if i == 1:
        dummy_teleport(env=env)
""") == """[True] 0
[True] 1
[False] 2
[False] 3
"""


def test_simple_ex_clause_0(env_getter):
    assert run_python(
f"""
{test_preamble}
{env_getter}

class CustomException(Exception):
    pass

log("try")
try:
    log("teleport")
    dummy_teleport(env=env)
    log("raise")
    raise CustomException("hello")
    log("unreachable")
except CustomException as e:
    log("handle")
log("done")
""") == """[True] try
[True] teleport
[False] raise
[False] handle
[False] done
"""


def test_simple_ex_clause_1(env_getter):
    assert run_python(
f"""
{test_preamble}
{env_getter}

class CustomException(Exception):
    pass

log("try")
try:
    log("teleport")
    dummy_teleport(env=env)
    log("raise")
    raise CustomException("hello")
    log("unreachable")
except CustomException as e:
    log("handle")
finally:
    log("finally")
log("done")
""") == """[True] try
[True] teleport
[False] raise
[False] handle
[False] finally
[False] done
"""


def test_simple_ex_clause_1_inside_finally(env_getter):
    assert run_python(
f"""
{test_preamble}
{env_getter}

class CustomException(Exception):
    pass

log("try")
try:
    log("raise")
    raise CustomException("hello")
    log("unreachable")
except CustomException as e:
    log("handle")
finally:
    log("teleport")
    dummy_teleport(env=env)
    log("finally")
log("done")
""") == """[True] try
[True] raise
[True] handle
[True] teleport
[False] finally
[False] done
"""


def test_ex_complex_stack(env_getter):
    assert run_python(
f"""
{test_preamble}
{env_getter}

class CustomException(Exception):
    pass

for j in range(3):
    log(f"loop {{j}}")
    if j == 0:
        log(f"try")
        try:
            for i in range(3, 6):
                log("teleport")
                dummy_teleport(env=env)
                log("raise")
                raise CustomException("hello")
            log("unreachable")
        except CustomException as e:
            log("handle")
        finally:
            log("finally")
log("done")
""") == """[True] loop 0
[True] try
[True] teleport
[False] raise
[False] handle
[False] finally
[False] loop 1
[False] loop 2
[False] done
"""


if __name__ == "__main__":
    import pytest
    pytest.main()

