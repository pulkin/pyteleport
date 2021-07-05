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
    dump.seek(0)
    assert run_python(
f"""
from flow_control import load
load("{dump.name}")
""") == 'world\n'


def test_nested():
    assert run_python("""
from flow_control import snapshot
flag = []
def a():
    def b():
        def c():
            flag.append("entered")
            result = "hello"
            snapshot(None)
            flag.append("exited")
            return result + " world"
        return len(c()) + float("3.5")
    return 5 * (3 + b())
state = a()
assert flag == ["entered"]
morph = state.compose_morph()
assert morph() == 87.5
assert flag == ["entered", "exited"]
""") == ""

if __name__ == "__main__":
    import pytest
    pytest.main()

