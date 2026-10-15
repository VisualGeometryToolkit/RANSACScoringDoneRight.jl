#!/bin/bash
# EXAMPLE SLURM launcher: the resources below are those of the cluster the
# published numbers were measured on (1900 GB / 224 CPU nodes in a partition
# named `gpu`). Adapt --partition, CPUs and memory to your site. Submit from the
# repository root (`sbatch neurips2026/scripts/slurm/<name>.sbatch`); logs go to ./logs/.
# Data paths come from the environment: CAU_DATA_ROOT, or POOL/OUT directly.
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --job-name=sb-scoring-miscal
#SBATCH --output=logs/sb_scoring_miscal_%j.log
#SBATCH --error=logs/sb_scoring_miscal_%j.log
set -euo pipefail
# MEMORY: --mem-per-cpu=8G is the node ratio (1900 GB / 224 CPU = 8.5 GB/CPU);
# cpus-per-task follows from peak memory / 8G. 32 CPUs => 256 GB.
# Job 4896 was OOM-killed at 300 GB after ~80 min: the driver holds one scene's
# `pairs` plus, per thread, a score buffer and a curves row, and the largest
# PT-E-SIFT-4K_pf3 scenes are ~98 GB on disk and expand well past that in memory.
# The driver now frees each scene before loading the next (GC.gc(true)), which
# halves the peak: without it two ~125 GB scenes were resident at once. The
# driver logs VmHWM per scene -- check it before changing this number.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

# =============================================================================
# sample_budget_miscal.sbatch — scoring-only sample-budget sweep, ζ ∈ {1, 32}.
#
# MEMORY: the driver holds ONE scene's `pairs` at a time. PT-E-SIFT-4K_pf3 scenes
# are 60–100 GB each (4096 models/pair), so the 900G request is sized to the
# largest single scene with headroom, NOT to the whole pool. The scene binding
# is loop-local so each scene is collectable before the next is read; do not
# refactor the loop to accumulate pools across scenes.
#
# CPU-only: no GPU is used (the sweep reads stored canonical Sampson distances
# and never re-solves). It requests the gpu partition purely because that is
# where the large-memory nodes live.
# =============================================================================

POOL=${POOL:-${CAU_DATA_ROOT:?set CAU_DATA_ROOT (or POOL and OUT)}/accv2026/data/PT-E-SIFT-4K_pf3}
RANGES=${RANGES:-$POOL/coarse_ranges_hierarchical.jld2}
OUT=${OUT:-${CAU_DATA_ROOT:?set CAU_DATA_ROOT (or POOL and OUT)}/accv2026/results/sample_budget_miscal}
# Replay-derived minimal-sample starts of the pool (accv2026
# scripts/tools/derive_pool_samples.jl): PT-E-SIFT-4K_pf3's record does not
# certify its own sample boundaries (duplicated E roots), so it needs them.
POOL_SAMPLES=${POOL_SAMPLES:-}
ZETAS=${ZETAS:-1,32}
CRITERION=${CRITERION:-median}

# The scenes: the TEST side of a registered split of the stack's one scene-split
# registry (VisualGeometryEval configs/splits.toml) — the validation side
# tuned the natives being consumed, so reading the budget curves off it would be
# tuning on the test set. No scene list lives here.
SPLITS=${SPLITS:-${VGEVAL_DIR:?set VGEVAL_DIR to a VisualGeometryEval.jl checkout (or SPLITS)}/configs/splits.toml}
SPLIT=${SPLIT:-phototourism}

echo "[$(date +%T)] pool=$POOL"
echo "[$(date +%T)] ranges=$RANGES"
echo "[$(date +%T)] splits=$SPLITS  split=$SPLIT (test scenes)"
echo "[$(date +%T)] out=$OUT  zetas=$ZETAS  criterion=$CRITERION"

exec julia -t "${SLURM_CPUS_PER_TASK:-auto}" --startup-file=no \
    --project=neurips2026 neurips2026/scripts/studies/sample_budget_miscal.jl \
    --pool="$POOL" --ranges="$RANGES" --out="$OUT" \
    --zetas="$ZETAS" --criterion="$CRITERION" --splits="$SPLITS" --split="$SPLIT" \
    --pool-samples="$POOL_SAMPLES"
