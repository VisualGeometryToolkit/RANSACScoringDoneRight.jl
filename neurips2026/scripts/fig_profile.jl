#!/usr/bin/env julia
# provenance: every product writes _config.toml and _provenance.toml (produce) beside its renders.
# =============================================================================
# scripts/fig_profile.jl — Fig. 1 (\label{fig:profile-grid}, "Profile"): the
# three Profile cells (PT/E, ETH3D/F, HEB/H) and the Profile style sheets,
# drawn from the stored runs of record (configs/fig_profile.toml).
#
# reproduce: reads configs/fig_profile.toml
# reproduce: writes paper/figures/PT-E-SIFT-Profile.tex
# reproduce: writes paper/figures/ETH3D-F-SPLG-Profile.tex
# reproduce: writes paper/figures/HEB-H-SIFT-Profile.tex
# reproduce: writes paper/figures/profile_styles.tex
# reproduce: writes paper/figures/profile_styles_shared.tex
#
# STAGES
#   render  each cell's fragment and the two style sheets from its stored
#           `source.jld2` (src/panels.jl, the port of the submitted renderer):
#           the keyed product → <results>/figures/fig_profile/<key>/
#   verify  each render against the submitted paper/figures copy, byte for
#           byte and by coordinates → verify.tsv beside the renders
#   deploy  into paper/figures — only where that changes nothing, unless
#           [deploy] replace_submitted (the author's sign-off on a difference)
#
#   julia --project=. scripts/fig_profile.jl [--plan] [--stage=NAME] [--force]
# =============================================================================
using DrWatson
@quickactivate "NeurIPS2026"
include(srcdir("NeurIPS2026.jl"))

const CFG_PATH = projectdir("configs", "fig_profile.toml")
const CFG = TOML.parsefile(CFG_PATH)
const CELLS = CFG["profile"]["cells"]
const ASSETS = vcat([c["asset"] for c in CELLS], [CFG["render"]["styles"], CFG["render"]["styles_shared"]])
renders() = figure_product("fig_profile", CFG)       # the keyed renders (src/assets.jl)

function render(dir)
    crit = CFG["profile"]["criterion"]
    styles = nothing
    for c in CELLS
        records = profile_records(JLD2.load(input_path(CFG, c["input"]), "profile"), crit)
        isempty(records) && error("no Profile records for criterion $crit in input $(c["input"])")
        oracle_y, pmb = profile_references(JLD2.load(input_path(CFG, c["input"]), "profile_oracle"), crit)
        open(joinpath(dir, c["asset"]), "w") do io
            write_profile_group(io, records, c["title"]; oracle_y, per_method_best = pmb, bands = c["bands"])
        end
        c["key"] == CFG["profile"]["styles_from"] && (styles = records)
    end
    styles === nothing && error("[profile] styles_from = $(repr(CFG["profile"]["styles_from"])) names no cell")
    return write_style_sheets((s, sh) -> write_profile_styles(s, sh, styles, crit), dir, CFG)
end

const STAGES = [
    Stage(:render, "the three Profile fragments and the style sheets, keyed by the config and the code";
          items = () -> [renders()], product = identity,
          run = p -> produce(render, p; driver = "fig_profile.jl", config_source = CFG_PATH,
                             inputs = [input_path(CFG, c["input"]) for c in CELLS]));
    asset_stages(renders, CFG, ASSETS)...
]

run_stages("fig_profile", STAGES)
