from subprocess import check_output, Popen, PIPE
import sys
from pathlib import Path
import ast
from itertools import product

import pytest


python_version = sys.version_info.major * 0x100 + sys.version_info.minor


def run_test(name, stack_method="inject", interactive=False):
    if interactive:
        with open(name, 'r') as fl:
            process = Popen([sys.executable], stdin=PIPE, stdout=PIPE, stderr=PIPE, encoding='utf-8')
            stdout, stderr = process.communicate(f"""
import sys
sys.argv = [*sys.argv, "stack_method={stack_method}"]
{fl.read()}
""")
            assert process.returncode == 0
            return stdout
    else:
        return check_output([
            sys.executable,
            name,
            f"stack_method={stack_method}"
        ], stderr=PIPE, text=True, env={"PYTHONPATH": "."})


test_cases = list(map(lambda x: x.name, (Path(__file__).parent / "tests").glob("_test_teleport_*.py")))
test_cases = list(product(test_cases, ["inject", "predict"], [False, True]))
# 'finally' clause is not recovered by 'predict' method in py3.8 and earlier
for interactive in (False, True):
    param = ("_test_teleport_finally.py", "predict", interactive)
    x = test_cases.index(param)
    test_cases[x] = pytest.param(*param, marks=pytest.mark.skipif(
        python_version <= 0x0308,
        reason="'finally' clause is not recovered by 'predict' method in py3.8 and earlier"))


@pytest.mark.parametrize("test, stack_method, interactive", test_cases)
def test_external(test, stack_method, interactive):
    test = Path(__file__).parent / "tests" / test

    with open(test, 'r') as f:
        module_text = f.read()
        module = ast.parse(module_text)
        docstring = ast.get_docstring(module)

    assert run_test(test, stack_method, interactive).rstrip() == eval(f'f"""{docstring}"""')
