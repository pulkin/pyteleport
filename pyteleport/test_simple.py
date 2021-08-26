from subprocess import check_output
from tempfile import NamedTemporaryFile

import pytest


def run_python(script):
    return check_output(["python"], input=script, text=True)


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

def test_simple_teleport(env_getter):
    assert run_python(
f"""
from pyteleport import dummy_teleport
import os
{env_getter}

parent_pid = os.getpid()

def log(*args):
    print(f"[{{os.getpid() == parent_pid}}]", *args, flush=True)

log("hello")
dummy_teleport(env=env)
log("world")
""") == "[True] hello\n[False] world\n"

def test_nested_teleport(env_getter):
    assert run_python(
f"""
from pyteleport import dummy_teleport
{env_getter}

def a():
    def b():
        def c():
            print("entered", flush=True)
            result = "hello"
            r2 = "world"
            dummy_teleport(env=env)
            assert result == "hello"
            assert r2 == "world"
            print("exited")
            return result + " world"
        return len(c()) + float("3.5")
    return 5 * (3 + b())
assert a() == 87.5
print("OK")
""") == "entered\nexited\nOK\n"

def test_nested_teleport_w_globals(env_getter):
    assert run_python(
f"""
from pyteleport import dummy_teleport
import os
{env_getter}

parent_pid = os.getpid()

def log(*args):
    print(f"[{{os.getpid() == parent_pid}}]", *args, flush=True)

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
from pyteleport import dummy_teleport
import os
{env_getter}

parent_pid = os.getpid()
generator = iter(range(2))

def log(*args):
    print(f"[{{os.getpid() == parent_pid}} {{next(generator)}}]", *args, flush=True)

log("hello")
dummy_teleport(env=env)
log("world")
""") == "[True 0] hello\n[False 1] world\n"

def test_simple_teleport_builtin_count(env_getter):
    assert run_python(
f"""
from pyteleport import dummy_teleport
import os
from itertools import count
{env_getter}

parent_pid = os.getpid()
generator = iter(count())

def log(*args):
    print(f"[{{os.getpid() == parent_pid}} {{next(generator)}}]", *args, flush=True)

log("hello")
dummy_teleport(env=env)
log("world")
""") == "[True 0] hello\n[False 1] world\n"


def test_simple_teleport_generator(env_getter):
    assert run_python(
f"""
from pyteleport import dummy_teleport
import os
{env_getter}

parent_pid = os.getpid()
def generator_fn():
    yield 0
    yield 1
generator = generator_fn()

def log(*args):
    print(f"[{{os.getpid() == parent_pid}} {{next(generator)}}]", *args, flush=True)

log("hello")
dummy_teleport(env=env)
log("world")
""") == "[True 0] hello\n[False 1] world\n"


def test_simple_loop(env_getter):
    assert run_python(
f"""
from pyteleport import dummy_teleport
import os
{env_getter}

parent_pid = os.getpid()

def log(*args):
    print(f"[{{os.getpid() == parent_pid}}]", *args, flush=True)

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
from pyteleport import dummy_teleport
import os
{env_getter}

parent_pid = os.getpid()

def log(*args):
    print(f"[{{os.getpid() == parent_pid}}]", *args, flush=True)

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
from pyteleport import dummy_teleport
import os
{env_getter}

parent_pid = os.getpid()

def log(*args):
    print(f"[{{os.getpid() == parent_pid}}]", *args, flush=True)

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
from pyteleport import dummy_teleport
import os
{env_getter}

parent_pid = os.getpid()

def log(*args):
    print(f"[{{os.getpid() == parent_pid}}]", *args, flush=True)

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


if __name__ == "__main__":
    import pytest
    pytest.main()

