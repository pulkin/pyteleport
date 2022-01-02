import os
from subprocess import check_output, Popen, PIPE
import sys
from pathlib import Path
import ast

import pytest


def run_test(name, local_environ=False, stack_method="inject", interactive=False):
    if interactive:
        with open(name, 'r') as fl:
            process = Popen([sys.executable], stdin=PIPE, stdout=PIPE, stderr=PIPE, encoding='utf-8')
            stdout, stderr = process.communicate(f"""
import sys
sys.argv = [*sys.argv, "local_environ={local_environ}", "stack_method={stack_method}"]
{fl.read()}
""")
            assert process.returncode == 0
            return stdout
    else:
        environ = os.environ.copy()
        if local_environ:
            environ["PYTHONPATH"] = os.getcwd() + ":" + environ.get("PYTHONPATH", "")
        return check_output([
            sys.executable,
            name,
            f"local_environ={local_environ}",
            f"stack_method={stack_method}"
        ], stderr=PIPE, text=True, env=environ)


@pytest.fixture()
def local_environ(pytestconfig):
    return bool(pytestconfig.getoption("inherit_path"))


test_cases = list(map(lambda x: x.name, (Path(__file__).parent / "tests").glob("_test_teleport_*.py")))


@pytest.mark.parametrize("test", test_cases)
@pytest.mark.parametrize("stack_method", ["inject", "predict"])
@pytest.mark.parametrize("interactive", [False, True])
def test_external(local_environ, test, stack_method, interactive):
    test = Path(__file__).parent / "tests" / test

    with open(test, 'r') as f:
        module_text = f.read()
        module = ast.parse(module_text)
        docstring = ast.get_docstring(module)

    assert run_test(test, local_environ, stack_method, interactive).rstrip() == docstring
