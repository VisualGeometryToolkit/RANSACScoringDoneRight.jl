#!/bin/bash
# Every result and figure of the paper on the owner's server (Slurm): reproduce.sh with its site profile. --dry-run, --only=…, --no-deps, status.
exec bash "$(dirname "${BASH_SOURCE[0]}")/reproduce.sh" --site=dgx01 "$@"
