#!/bin/bash
# EXAMPLE SLURM launcher: the resources below are those of the cluster the
# published numbers were measured on (1900 GB / 224 CPU nodes in a partition
# named `gpu`). Adapt --partition, CPUs and memory to your site. Submit from the
# repository root (`sbatch neurips2026/scripts/slurm/<name>.sbatch`); logs go to ./logs/.
# Data paths come from the environment: CAU_DATA_ROOT, or POOL/OUT directly.
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=score-timing
#SBATCH --output=logs/score_timing_%j.log
#SBATCH --error=logs/score_timing_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

# =============================================================================
# score_timing.sbatch — THE launcher for the per-hypothesis scoring-cost table.
#
# Reproduces the published table exactly. The reported numbers came from this
# invocation; do not change --npairs or --reps without re-running everything,
# because the figure is a per-hypothesis minimum over repetitions and is only
# comparable across rows measured in the same process.
#
# SINGLE THREAD BY PROTOCOL. `julia -t 1` is not negotiable: the driver
# hard-errors if nthreads() != 1 rather than publishing a threaded number under
# a single-thread label. The 2 CPUs are for the Julia process plus GC, not for
# parallel scoring.
#
# MEMORY: --mem-per-cpu=8G, the node ratio (1900 GB / 224 CPU = 8.5 GB/CPU).
# NEVER use a flat --mem: reserving memory out of proportion to CPUs strands the
# rest of the node. british_museum.jld2 is ~92 GB and the load needs ~220 GB, so
# 28 CPUs is what that memory costs at the node ratio -- even though the scoring
# itself is single-threaded by protocol.
# =============================================================================

POOL=${POOL:-${CAU_DATA_ROOT:?set CAU_DATA_ROOT (or POOL and OUT)}/accv2026/data/PT-E-SIFT-4K}
OUT=${OUT:-${CAU_DATA_ROOT:?set CAU_DATA_ROOT (or POOL and OUT)}/accv2026/results/score_timing}
SCENE=${SCENE:-british_museum}
NPAIRS=${NPAIRS:-20}
REPS=${REPS:-3}
LOSS_WIDTH=${LOSS_WIDTH:-0.00164}

echo "[$(date +%T)] scene=$SCENE pool=$POOL npairs=$NPAIRS reps=$REPS loss_width=$LOSS_WIDTH"

exec julia -t 1 --startup-file=no --project=neurips2026 \
    neurips2026/scripts/studies/score_timing.jl \
    --pool="$POOL" --out="$OUT" --scene="$SCENE" \
    --npairs="$NPAIRS" --reps="$REPS" --loss-width="$LOSS_WIDTH"
