# =============================================================================
# src/assets.jl — the last two stages of every figure driver: VERIFY a
# regenerated paper asset against the submitted one, and DEPLOY it into
# paper/figures only when that changes nothing or the author signed off.
#
# A driver renders into its keyed results directory (never into the paper);
# `verify_assets` compares each file with the paper's copy — byte for byte, and
# by its coordinates (`compare_fragments`) — and writes the verdict to
# `verify.tsv` beside the renders; `deploy_assets` then copies a render into
# paper/figures when the paper lacks it or holds the identical bytes, and
# REFUSES when they differ unless the config's [deploy] replace_submitted is
# true. A submitted asset is never silently replaced.
# =============================================================================

"The directory the paper reads its generated assets from."
paper_figdir() = projectdir("paper", "figures")

"The stored runs of record a figure config names ([stored] dir, relative to results_root())."
stored_dir(cfg::AbstractDict) = joinpath(results_root(), cfg["stored"]["dir"])

"The evaluated table `key` of a stored run (`run`, relative to the stored directory)."
function stored_table(cfg::AbstractDict, run::AbstractString, key::AbstractString)
    src = require_input(joinpath(stored_dir(cfg), run, "source.jld2");
        what = "the evaluated table of stored run $run",
        provenance = "the 2026-05-06 production runs (PROVENANCE.toml beside them); fetch the neurips2026_paper_runs artifact (data/README.md)")
    return JLD2.load(src, key)
end

const VERIFY_FILE = "verify.tsv"

"""
    verify_assets(outdir, names) -> Vector{NamedTuple}

Compare each rendered `outdir/<name>` with `paper/figures/<name>` and write
`outdir/verify.tsv` (name, identical, same_numbers, coordinate blocks, largest
absolute difference). A missing paper copy is reported, not an error.
"""
function verify_assets(outdir::AbstractString, names)
    rows = NamedTuple[]
    for n in names
        new = read(joinpath(outdir, n), String)
        old_path = joinpath(paper_figdir(), n)
        if isfile(old_path)
            c = compare_fragments(new, read(old_path, String))
            push!(rows, (; name = String(n), submitted = true, c...))
        else
            push!(rows, (name = String(n), submitted = false, identical = false, same_numbers = false,
                         blocks_new = length(fragment_coordinates(new)), blocks_old = 0, max_abs_diff = NaN))
        end
    end
    open(joinpath(outdir, VERIFY_FILE), "w") do io
        println(io, join(("asset", "submitted", "identical", "same_numbers", "blocks_new", "blocks_old", "max_abs_diff"), '\t'))
        for r in rows
            println(io, join((r.name, r.submitted, r.identical, r.same_numbers, r.blocks_new, r.blocks_old, r.max_abs_diff), '\t'))
        end
    end
    for r in rows
        verdict = !r.submitted ? "NEW (no submitted copy)" : r.identical ? "identical" :
                  r.same_numbers ? "same numbers, formatting differs" :
                  "DIFFERS (max |Δ| = $(r.max_abs_diff), blocks $(r.blocks_new) vs $(r.blocks_old))"
        println("   ", rpad(r.name, 40), verdict)
    end
    return rows
end

"Whether every rendered asset is byte-identical to the paper's copy."
assets_deployed(outdir::AbstractString, names) =
    all(n -> isfile(joinpath(outdir, n)) && isfile(joinpath(paper_figdir(), n)) &&
             read(joinpath(outdir, n)) == read(joinpath(paper_figdir(), n)), names)

"""
    deploy_assets(outdir, names, cfg)

Copy each rendered asset into paper/figures when the paper has no copy or the
identical one; a DIFFERENT submitted copy is replaced only when the config's
[deploy] replace_submitted is true, and refused otherwise (the verify stage's
`verify.tsv` states the difference for the author's sign-off).
"""
function deploy_assets(outdir::AbstractString, names, cfg::AbstractDict)
    replace_submitted = get(get(cfg, "deploy", Dict()), "replace_submitted", false) === true
    refused = String[]
    for n in names
        new, dst = joinpath(outdir, n), joinpath(paper_figdir(), n)
        if isfile(dst) && read(new) == read(dst)
            continue
        elseif isfile(dst) && !replace_submitted
            push!(refused, n)
        else
            mkpath(paper_figdir())
            cp(new, dst; force = true)
            println("   deployed $n")
        end
    end
    isempty(refused) || error("deploy: the regenerated $(join(refused, ", ")) differ(s) from the submitted asset(s); " *
        "see $(joinpath(outdir, VERIFY_FILE)). Set [deploy] replace_submitted = true in the config once the difference is signed off.")
    return nothing
end
