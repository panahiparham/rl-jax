#!/usr/bin/env python3
"""Bootstrap the SLURM cluster for running experiments.

usage: setup_cluster.py [--config PATH]

All the logic lives in experiment.slurm.setup(): the bare repo, the shared
venvs, and the account-candidate report.
"""

import argparse

from experiment import slurm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", help="cluster/slurm configuration file (default cluster.toml)"
    )
    args = parser.parse_args()
    slurm.setup(config_path=args.config)


if __name__ == "__main__":
    main()
