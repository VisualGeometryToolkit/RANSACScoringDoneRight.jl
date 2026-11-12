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
# holds only where the paper is. (A config's inputs are read with Eval's
# `input_path(cfg, id)`: the tier's root, the sha256 checked.)
# =============================================================================
using VisualGeometryToolkitEval.Reproduce: verify_assets, assets_deployed, deploy_assets,
    fragment_coordinates, compare_fragments, VERIFY_REPORT
using VisualGeometryToolkitEval.PaperProjects: paper_figures_dir   # paper/figures/: where the paper reads its generated assets


"""
    verify_paper_assets(outdir, assets)
    deploy_paper_assets(outdir, cfg, assets)

The verify and deploy stages of every driver: `verify_assets` and
`deploy_assets` of the renders `assets` in `outdir` against paper/figures
(`paper_figures_dir()`), the deploy under the config's sign-off
(`replace_submitted(cfg)`).
"""
verify_paper_assets(outdir::AbstractString, assets) = (verify_assets(outdir, paper_figures_dir(), assets); nothing)
deploy_paper_assets(outdir::AbstractString, cfg::AbstractDict, assets) =
    (deploy_assets(outdir, paper_figures_dir(), assets; replace_submitted = replace_submitted(cfg)); nothing)

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
