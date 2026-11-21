#!/bin/bash
# Every result and figure of the paper on the vcai cluster (Slurm): reproduce.sh with its site profile, whose [storage] names
# the data and results roots and the artifact store on the cluster's project space. --dry-run, --only=…, --no-deps, status.
exec bash "$(dirname "${BASH_SOURCE[0]}")/reproduce.sh" --site=vcai "$@"
