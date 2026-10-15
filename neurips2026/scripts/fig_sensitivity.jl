#!/usr/bin/env julia
# provenance: every run writes _config.toml and _provenance.toml (write_run_records) beside its renders.
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
#           renderer) → <results>/figures/fig_sensitivity-<run key>/
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
const KEY = run_key(CFG)                       # [render] is excluded
const OUTDIR = joinpath(results_root(), "figures", "fig_sensitivity-" * KEY)
const SENS = CFG["sensitivity"]
const CELLS = SENS["cells"]
const ASSETS = vcat([c["assets"][p] for c in CELLS for p in SENS["panels"]],
                    [CFG["render"]["styles"], CFG["render"]["styles_shared"]])

function render()
    mkpath(OUTDIR)
    inputs = String[]
    for c in CELLS
        data = sensitivity_data(stored_table(CFG, c["run"], "summary"))
        for p in SENS["panels"]
            open(io -> write_sensitivity_group(io, sensitivity_panel(p), data, SENS["tune"], c["title"]),
                 joinpath(OUTDIR, c["assets"][p]), "w")
        end
        if c["key"] == SENS["styles_from"]
            open(joinpath(OUTDIR, CFG["render"]["styles"]), "w") do s
                open(sh -> write_sensitivity_styles(s, sh, data), joinpath(OUTDIR, CFG["render"]["styles_shared"]), "w")
            end
        end
        push!(inputs, joinpath(stored_dir(CFG), c["run"], "source.jld2"))
    end
    write_run_records(OUTDIR, CFG; driver = "fig_sensitivity.jl", config_source = CFG_PATH, key = KEY, inputs)
    return nothing
end

verify() = (verify_assets(OUTDIR, ASSETS); nothing)
deploy() = deploy_assets(OUTDIR, ASSETS, CFG)

const STAGES = [
    Stage(:render, "the six Sensitivity panels and the style sheets, keyed by the run key",
          () -> run_complete(OUTDIR; sidecar = first(ASSETS)) && all(a -> isfile(joinpath(OUTDIR, a)), ASSETS), render),
    Stage(:verify, "each render against the submitted asset (bytes, coordinates)",
          () -> fresher(joinpath(OUTDIR, VERIFY_FILE), (joinpath(OUTDIR, a) for a in ASSETS)...), verify),
    Stage(:deploy, "into paper/figures where nothing changes (or signed off)", () -> assets_deployed(OUTDIR, ASSETS), deploy),
]

run_stages("fig_sensitivity", STAGES)
