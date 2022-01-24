from subprocess import check_output, Popen, PIPE
from tempfile import NamedTemporaryFile
import sys

import pytest


def run_python(script=None, debug=False):
    if script is None:
        return Popen([sys.executable], stdin=PIPE, stdout=PIPE, stderr=PIPE, encoding='utf-8')
    else:
        if debug:
            print(script)
        return check_output([sys.executable], input=script, text=True)


def test_trivial():
    dump = NamedTemporaryFile()
    assert run_python(
f"""
from pyteleport.snapshot import dump
print('hello', flush=True)
dump(open("{dump.name}", 'wb'))
print('world')
""") == 'hello\n'
    assert run_python(
f"""
from pyteleport.snapshot import load
load(open("{dump.name}", 'rb'))()
""") == 'world\n'


def test_nested():
    dump = NamedTemporaryFile()
    assert run_python(
f"""
from pyteleport.snapshot import dump
def a():
    def b():
        def c():
            print("entered", flush=True)
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
from pyteleport.snapshot import load
load(open("{dump.name}", 'rb'))()
""") == 'exited\nOK\n'
