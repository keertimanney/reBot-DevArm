from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .pinocchio_meshcat import UPSTREAM_REPO, default_control_repo, find_urdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the B601 Pinocchio/MeshCat simulation.")
    parser.add_argument("--control-repo", type=Path, default=default_control_repo())
    parser.add_argument("--fetch-upstream", action="store_true", help="Clone the upstream URDF/sim repo if missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    control_repo = args.control_repo.expanduser().resolve()

    if not control_repo.exists():
        if not args.fetch_upstream:
            raise SystemExit(
                f"{control_repo} does not exist. Re-run with --fetch-upstream or clone {UPSTREAM_REPO} there."
            )
        control_repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", UPSTREAM_REPO, str(control_repo)], check=True)

    urdf = find_urdf(control_repo)
    print(f"Control repo: {control_repo}")
    print(f"URDF: {urdf}")
    print("Install this package with: python -m pip install -e rebot_b601_sim")
    print("Run simulation with: rebot-b601-sim")


if __name__ == "__main__":
    main()

