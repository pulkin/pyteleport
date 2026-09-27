import dis

import pytest

from pyteleport.bytecode.exceptiontable import pack_exception_table, unpack_exception_table
from pyteleport.bytecode.opcodes import python_feature_exceptiontable
from pyteleport.bytecode.primitives import ExceptionCodeBlock


pytestmark = pytest.mark.skipif(
    not python_feature_exceptiontable,
    reason="exception tables require Python 3.11+",
)


@pytest.mark.parametrize("source", [
    "try:\n    call()\nexcept Exception:\n    handle()",
    "try:\n    call()\nfinally:\n    cleanup()",
    "try:\n" + "    call()\n" * 100 + "finally:\n    cleanup()",
])
def test_exception_table_round_trip(source):
    code = compile(source, "", "exec")

    assert pack_exception_table(unpack_exception_table(code)) == code.co_exceptiontable
