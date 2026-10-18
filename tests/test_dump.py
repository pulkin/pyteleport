import subprocess
import sys


def test_dump_writes_resumable_pyc(tmp_path):
    payload = tmp_path / "snapshot.pyc"
    script = tmp_path / "create_snapshot.py"
    script.write_text(
        "from pyteleport import dump\n"
        f"x = dump({str(payload)!r}, tos=lambda: 42, call_tos=True)\n"
        "print('resumed', x)\n"
    )

    subprocess.run([sys.executable, str(script)], check=True, capture_output=True, text=True)
    result = subprocess.run([sys.executable, str(payload)], check=True, capture_output=True, text=True)

    assert result.stdout == "resumed 42\n"
