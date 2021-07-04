from subprocess import check_output, STDOUT
from tempfile import NamedTemporaryFile


def run_python(script):
    return check_output(["python"], input=script, text=True, stderr=STDOUT, shell=True)


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


test_trivial()

