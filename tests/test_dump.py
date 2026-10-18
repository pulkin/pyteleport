import subprocess
import sys


def test_dump_writes_resumable_pyc(tmp_path):
    payload = tmp_path / "snapshot.pyc"
    script = tmp_path / "create_snapshot.py"
    script.write_text(
        "from pyteleport import dump\n"
        f"print('resumed', (dump({str(payload)!r}, return_=lambda: 42) or (lambda: None))())\n"
    )

    subprocess.run([sys.executable, str(script)], check=True, capture_output=True, text=True)
    result = subprocess.run([sys.executable, str(payload)], check=True, capture_output=True, text=True)

    assert result.stdout == "resumed 42\n"
