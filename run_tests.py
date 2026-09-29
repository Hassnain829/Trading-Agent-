"""
Run every test suite in tests/ and print one line per suite.

    .venv\\Scripts\\python.exe run_tests.py            # all suites
    .venv\\Scripts\\python.exe run_tests.py scalper    # only suites whose name contains "scalper"

Each suite is a plain script that prints PASS/FAIL lines and exits non-zero on failure.
They sandbox every data file in a temp folder, and none of them needs MT5 or the network.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv: list) -> int:
    pattern = argv[0] if argv else ""
    suites = sorted(p for p in (ROOT / "tests").glob("test_*.py") if pattern in p.stem)
    if not suites:
        print(f"no test suites match {pattern!r}")
        return 1
    env = {**os.environ, "PYTHONUTF8": "1"}
    failed = []
    started = time.monotonic()
    for suite in suites:
        t0 = time.monotonic()
        result = subprocess.run([sys.executable, str(suite)], cwd=ROOT, env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        verdict = lines[-1] if lines else "(no output)"
        ok = result.returncode == 0
        print(f"{'OK  ' if ok else 'FAIL'} {suite.stem:26} {time.monotonic() - t0:5.1f}s  {verdict}")
        if not ok:
            failed.append(suite.stem)
            for line in [l for l in lines if l.startswith("FAIL")][:8]:
                print(f"       {line[:200]}")
            if result.returncode != 0 and not any(l.startswith("FAIL") for l in lines):
                print("       " + "\n       ".join(result.stderr.strip().splitlines()[-6:]))
    print(f"\n{len(suites) - len(failed)}/{len(suites)} suites passed in {time.monotonic() - started:.0f}s"
          + (f" | failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
