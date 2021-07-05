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


if __name__ == "__main__":
    import pytest
    pytest.main()

