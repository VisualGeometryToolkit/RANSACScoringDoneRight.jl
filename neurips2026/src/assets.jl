# =============================================================================
# src/assets.jl — the paper wiring of the last two stages of every figure
# driver: VERIFY a regenerated paper asset against the submitted one, and
# DEPLOY it into paper/figures only when that changes nothing or the author
# signed off.
#
# The stages themselves are VisualGeometryToolkitEval's (`Reproduce`):
# `verify_assets(outdir, paperdir, names)` compares each render with the
# paper's copy — byte for byte, and a `.tex` fragment by its numbers
# (`compare_fragments`) — and writes the verdict to `verify.tsv`
# (`VERIFY_REPORT`) beside the renders; `deploy_assets(outdir, paperdir, names;
# replace_submitted)` copies a render into paper/figures when the paper lacks
# it or holds the identical bytes, and REFUSES when they differ unless the
# config's [deploy] replace_submitted signs the difference off (true, or the
# names of the assets). A submitted asset is never silently replaced. This file
# holds only where the paper and the stored runs are.
# =============================================================================
using VisualGeometryToolkitEval.Reproduce: verify_assets, assets_deployed, deploy_assets,
    fragment_coordinates, compare_fragments, VERIFY_REPORT

"The directory the paper reads its generated assets from."
paper_figdir() = projectdir("paper", "figures")

"""
    replace_submitted(cfg) -> Bool | Vector{String}

The author's sign-off of a figure config, its [deploy] replace_submitted:
`true` for every asset, the names of the signed-off assets, or `false` (the
default) — the `replace_submitted` keyword of `deploy_assets`.
"""
function replace_submitted(cfg::AbstractDict)
    v = get(get(cfg, "deploy", Dict()), "replace_submitted", false)
    return v isa AbstractVector ? String.(v) : v
end

"The stored runs of record a figure config names ([stored] dir, relative to results_root())."
stored_dir(cfg::AbstractDict) = joinpath(results_root(), cfg["stored"]["dir"])

"The evaluated table `key` of a stored run (`run`, relative to the stored directory)."
function stored_table(cfg::AbstractDict, run::AbstractString, key::AbstractString)
    src = require_input(joinpath(stored_dir(cfg), run, "source.jld2");
        what = "the evaluated table of stored run $run",
        provenance = "the 2026-05-06 production runs (PROVENANCE.toml beside them); fetch the neurips2026_paper_runs artifact (data/README.md)")
    return JLD2.load(src, key)
end
