#!/usr/bin/env julia
# provenance: every product writes _config.toml and _provenance.toml (produce) beside its renders.
# =============================================================================
# scripts/fig_sensitivity.jl — Fig. 2 (\label{fig:sensitivity-grid},
# "Sensitivity"): the six panels (expected test rotation error and its std
# over the validation-set size; PT/E, ETH3D/F, HEB/H) and the Sensitivity
# style sheets, drawn from the stored runs of record
# (configs/fig_sensitivity.toml).
#
# reproduce: reads configs/fig_sensitivity.toml
# reproduce: writes paper/figures/PT-E-SIFT-Sensitivity-eRmean.tex
# reproduce: writes paper/figures/PT-E-SIFT-Sensitivity-eRstd.tex
# reproduce: writes paper/figures/ETH3D-F-SPLG-Sensitivity-eRmean.tex
# reproduce: writes paper/figures/ETH3D-F-SPLG-Sensitivity-eRstd.tex
# reproduce: writes paper/figures/HEB-H-SIFT-Sensitivity-eRmean.tex
# reproduce: writes paper/figures/HEB-H-SIFT-Sensitivity-eRstd.tex
# reproduce: writes paper/figures/sensitivity_styles.tex
# reproduce: writes paper/figures/sensitivity_styles_shared.tex
#
# STAGES
#   render  each cell's panels and the two style sheets from its stored
#           `source.jld2` `summary` (src/panels.jl, the port of the submitted
#           renderer): the keyed product → <results>/figures/fig_sensitivity/<key>/
#   verify  each render against the submitted paper/figures copy, byte for
#           byte and by coordinates → verify.tsv beside the renders
#   deploy  into paper/figures — only where that changes nothing, unless
#           [deploy] replace_submitted (the author's sign-off on a difference)
#
#   julia --project=. scripts/fig_sensitivity.jl [--plan] [--stage=NAME] [--force]
# =============================================================================
using DrWatson
@quickactivate "NeurIPS2026"
include(srcdir("NeurIPS2026.jl"))

const CFG_PATH = projectdir("configs", "fig_sensitivity.toml")
const CFG = TOML.parsefile(CFG_PATH)
const SENS = CFG["sensitivity"]
const CELLS = SENS["cells"]
const ASSETS = vcat([c["assets"][p] for c in CELLS for p in SENS["panels"]],
                    [CFG["render"]["styles"], CFG["render"]["styles_shared"]])
renders() = figure_product("fig_sensitivity", CFG)       # the keyed renders (src/assets.jl)

function render(dir)
    styles = nothing
    for c in CELLS
        data = results_by_method(JLD2.load(input_path(CFG, c["input"]), "summary"))
        for p in SENS["panels"]
            open(io -> write_sensitivity_group(io, sensitivity_panel(p), data, SENS["tune"], c["title"]),
                 joinpath(dir, c["assets"][p]), "w")
        end
        c["key"] == SENS["styles_from"] && (styles = data)
    end
    styles === nothing && error("[sensitivity] styles_from = $(repr(SENS["styles_from"])) names no cell")
    return write_style_sheets((s, sh) -> write_sensitivity_styles(s, sh, styles), dir, CFG)
end

const STAGES = [
    Stage(:render, "the six Sensitivity panels and the style sheets, keyed by the config and the code";
          items = () -> [renders()], product = identity,
          run = p -> produce(render, p; driver = "fig_sensitivity.jl", config_source = CFG_PATH,
                             inputs = [input_path(CFG, c["input"]) for c in CELLS]));
    asset_stages(renders, CFG, ASSETS)...
]

run_stages("fig_sensitivity", STAGES)
