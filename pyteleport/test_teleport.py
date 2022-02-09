from subprocess import check_output, Popen, PIPE
import sys
from pathlib import Path
import ast

import pytest


def run_test(name, interactive=False, timeout=2):
    if interactive:
        with open(name, 'r') as fl:
            process = Popen([sys.executable], stdin=PIPE, stdout=PIPE, stderr=PIPE, encoding='utf-8')
            stdout, stderr = process.communicate(fl.read(), timeout=2)
            assert process.returncode == 0
            return stdout
    else:
        return check_output([
            sys.executable,
            name,
        ], stderr=PIPE, text=True, env={"PYTHONPATH": "."}, timeout=timeout)


test_cases = list(map(lambda x: x.name, (Path(__file__).parent / "tests").glob("_test_teleport_*.py")))


@pytest.mark.parametrize("test", test_cases)
@pytest.mark.parametrize("interactive", [False, True])
def test_external(test, interactive):
    test = Path(__file__).parent / "tests" / test

    with open(test, 'r') as f:
        module_text = f.read()
        module = ast.parse(module_text)
        docstring = ast.get_docstring(module)

    assert run_test(test, interactive=interactive).rstrip() == eval(f'f"""{docstring}"""')
