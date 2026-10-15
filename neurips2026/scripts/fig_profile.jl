#!/usr/bin/env julia
# provenance: every run writes _config.toml and _provenance.toml (write_run_records) beside its renders.
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
#           `source.jld2` (src/panels.jl, the port of the submitted renderer)
#           → <results>/figures/fig_profile-<run key>/
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
const KEY = run_key(CFG)                       # [render] is excluded
const OUTDIR = joinpath(results_root(), "figures", "fig_profile-" * KEY)
const CELLS = CFG["profile"]["cells"]
const ASSETS = vcat([c["asset"] for c in CELLS], [CFG["render"]["styles"], CFG["render"]["styles_shared"]])

function render()
    crit = CFG["profile"]["criterion"]
    mkpath(OUTDIR)
    inputs = String[]
    for c in CELLS
        records = profile_records(stored_table(CFG, c["run"], "profile"), crit)
        isempty(records) && error("no Profile records for criterion $crit in $(c["run"])")
        oracle_y, pmb = profile_references(stored_table(CFG, c["run"], "profile_oracle"), crit)
        open(joinpath(OUTDIR, c["asset"]), "w") do io
            write_profile_group(io, records, c["title"]; oracle_y, per_method_best = pmb, bands = c["bands"])
        end
        if c["key"] == CFG["profile"]["styles_from"]
            open(joinpath(OUTDIR, CFG["render"]["styles"]), "w") do s
                open(sh -> write_profile_styles(s, sh, records, crit), joinpath(OUTDIR, CFG["render"]["styles_shared"]), "w")
            end
        end
        push!(inputs, joinpath(stored_dir(CFG), c["run"], "source.jld2"))
    end
    write_run_records(OUTDIR, CFG; driver = "fig_profile.jl", config_source = CFG_PATH, key = KEY, inputs)
    return nothing
end

verify() = (verify_assets(OUTDIR, ASSETS); nothing)
deploy() = deploy_assets(OUTDIR, ASSETS, CFG)

const STAGES = [
    Stage(:render, "the three Profile fragments and the style sheets, keyed by the run key",
          () -> run_complete(OUTDIR; sidecar = first(ASSETS)) && all(a -> isfile(joinpath(OUTDIR, a)), ASSETS), render),
    Stage(:verify, "each render against the submitted asset (bytes, coordinates)",
          () -> fresher(joinpath(OUTDIR, VERIFY_FILE), (joinpath(OUTDIR, a) for a in ASSETS)...), verify),
    Stage(:deploy, "into paper/figures where nothing changes (or signed off)", () -> assets_deployed(OUTDIR, ASSETS), deploy),
]

run_stages("fig_profile", STAGES)
