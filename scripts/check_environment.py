"""Check the software and solver requirements of the reported experiments."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys


EXPECTED = {
    "torch": "2.7.1",
    "transformers": "4.47.1",
    "numpy": "1.26.4",
    "scipy": "1.15.3",
    "cvxpy": "1.6.5",
    "clarabel": "0.11.1",
}


def installed_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if a CUDA device is not available to PyTorch.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    failures = []

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python: {python_version}")
    if sys.version_info[:2] != (3, 10):
        failures.append("Python 3.10 is required by the recorded environment")

    for distribution, expected in EXPECTED.items():
        observed = installed_version(distribution)
        print(f"{distribution}: {observed or 'not installed'}")
        if distribution == "torch" and observed:
            matches = observed.split("+", 1)[0] == expected
        else:
            matches = observed == expected
        if not matches:
            failures.append(
                f"{distribution} {expected} required, found {observed or 'none'}"
            )

    try:
        import cvxpy as cp

        solvers = set(cp.installed_solvers())
        print(f"CVXPY solvers: {', '.join(sorted(solvers)) or 'none'}")
        if "CLARABEL" not in solvers:
            failures.append("CVXPY does not report the CLARABEL solver")
    except Exception as exc:
        failures.append(f"CVXPY import or solver discovery failed: {exc}")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA runtime: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        elif args.require_cuda:
            failures.append("a CUDA device is required for formal training")
    except Exception as exc:
        failures.append(f"PyTorch import failed: {exc}")

    if failures:
        print("\nEnvironment check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
