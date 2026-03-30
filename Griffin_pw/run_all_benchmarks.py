"""
Master Benchmark Runner
=======================
Runs ALL benchmark suites sequentially.
Each individual script uses skip-if-exists logic: already-completed benchmarks
are skipped automatically, so this script is safe to re-run after a crash.

Results accumulate in:  results/complete_benchmark.json
Logs are saved in:      results/logs/

Benchmark suites
----------------
  run_5_benchmarks.py           Core LM benchmarks (MQAR, Induction Heads,
                                Copy, Selective Copy, Sequential NIAH)
  run_aan_benchmark.py          AAN forced long-range recall test
  run_classification_benchmarks.py  Classification tasks (Chomsky CFG,
                                    Path-X, ListOps)
  run_copy_seq_len.py           Copy task at seq_len 256 / 512 / 1024
  run_seq_scaling.py            MQAR + Induction Heads seq-length scaling
  run_mqar_breaking_point.py    MQAR gap-size ablation (gap 0 / 100 / 200)

Run from Griffin_pw/:
    python run_all_benchmarks.py

To run a single suite:
    python run_5_benchmarks.py
    python run_aan_benchmark.py
    python run_classification_benchmarks.py
    python run_copy_seq_len.py
    python run_seq_scaling.py
    python run_mqar_breaking_point.py
"""

import subprocess
import sys
import os
import time

# Each entry: (script_filename, short description)
BENCHMARK_SCRIPTS = [
    ("benchmarks/run_5_benchmarks.py",               "Core LM benchmarks (MQAR, Induction Heads, Copy, Selective Copy, Seq-NIAH)"),
    ("benchmarks/run_aan_benchmark.py",              "AAN forced long-range recall"),
    ("benchmarks/run_classification_benchmarks.py",  "Classification tasks (Chomsky, Path-X, ListOps)"),
    ("benchmarks/run_copy_seq_len.py",               "Copy at seq_len 256 / 512 / 1024"),
    ("benchmarks/run_seq_scaling.py",                "MQAR + Induction Heads seq-length scaling"),
    ("benchmarks/run_mqar_breaking_point.py",        "MQAR gap-size ablation"),
]


def banner(msg: str) -> None:
    line = "=" * 70
    print(f"\n{line}")
    print(f"  {msg}")
    print(f"{line}\n", flush=True)


def run_all() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    total = len(BENCHMARK_SCRIPTS)
    failures = []

    banner("Griffin / Hawk / Local-Attention Benchmark Suite")
    print(f"  Results file : results/complete_benchmark.json")
    print(f"  Log files    : results/logs/")
    print(f"  Total suites : {total}\n")

    for idx, (script, description) in enumerate(BENCHMARK_SCRIPTS, 1):
        script_path = os.path.join(root, script)
        if not os.path.exists(script_path):
            print(f"[{idx}/{total}] SKIP (not found): {script}")
            continue

        banner(f"[{idx}/{total}] {description}")
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, "-u", script_path],
            cwd=root,
        )
        elapsed = time.time() - t0

        status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        print(f"\n  -> {status}  [{elapsed/60:.1f} min]", flush=True)

        if result.returncode != 0:
            failures.append(script)

    # Final summary
    banner("Run complete")
    if failures:
        print(f"  {len(failures)} suite(s) failed:")
        for f in failures:
            print(f"    - {f}")
        sys.exit(1)
    else:
        print(f"  All {total} suites completed successfully.")
        print(f"  Results: results/complete_benchmark.json")


if __name__ == "__main__":
    run_all()
