from subprocess import check_output
from tempfile import NamedTemporaryFile
from .flow_control import save

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
from pyteleport.flow_control import save
print('hello')
save("{dump.name}")
print('world')
""") == 'hello\n'
    assert run_python(
f"""
from pyteleport.flow_control import load
load("{dump.name}")
""") == 'world\n'


def test_nested():
    dump = NamedTemporaryFile()
    assert run_python(
f"""
from pyteleport.flow_control import save
def a():
    def b():
        def c():
            print("entered")
            result = "hello"
            r2 = "world"
            save("{dump.name}")
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
from pyteleport.flow_control import load
load("{dump.name}")
""") == 'exited\nOK\n'


def test_nested_pack():
    dump = NamedTemporaryFile()
    assert run_python(
f"""
from pyteleport.flow_control import save
def a():
    def b():
        def c():
            print("entered")
            result = "hello"
            r2 = "world"
            save("{dump.name}", pack=True)
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
from pyteleport.flow_control import load
load("{dump.name}")
""") == 'exited\nOK\n'

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

if __name__ == "__main__":
    import pytest
    pytest.main()

