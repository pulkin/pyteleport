from subprocess import check_output
from tempfile import NamedTemporaryFile


def run_python(script):
    return check_output(["python"], input=script, text=True)


def test_trivial():
    dump = NamedTemporaryFile()
    assert run_python(
f"""
from flow_control import save
print('hello')
save("{dump.name}")
print('world')
""") == 'hello\n'
    assert run_python(
f"""
from flow_control import load
load("{dump.name}")
""") == 'world\n'


def test_nested():
    dump = NamedTemporaryFile()
    assert run_python(
f"""
from flow_control import save
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
from flow_control import load
load("{dump.name}")
""") == 'exited\nOK\n'


def test_nested_pack():
    dump = NamedTemporaryFile()
    assert run_python(
f"""
from flow_control import save
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
from flow_control import load
load("{dump.name}")
""") == 'exited\nOK\n'

def test_nested_teleport():
    dump = NamedTemporaryFile()
    assert run_python(
f"""
from flow_control import dummy_teleport
def a():
    def b():
        def c():
            print("entered", flush=True)
            result = "hello"
            r2 = "world"
            dummy_teleport()
            assert result == "hello"
            assert r2 == "world"
            print("exited")
            return result + " world"
        return len(c()) + float("3.5")
    return 5 * (3 + b())
assert a() == 87.5
print("OK")
""") == "entered\nexited\nOK\n"

if __name__ == "__main__":
    import pytest
    pytest.main()

