# =============================================================================
# src/numbers.jl — the statistics behind the numbers the prose states about
# the Profile and Sensitivity figures (configs/paper_numbers.toml states which
# number, from which stored run, under which rounding). Computed from the
# same evaluated tables the figures are drawn from (`input_path`).
# =============================================================================

# The rows of one method in a Sensitivity summary at the tuning criterion, by n.
function _sens_rows(summary::DataFrame, method::AbstractString, tune::AbstractString)
    r = summary[(summary.method .== method) .& (summary.tuning_criterion .== tune), :]
    isempty(r) && error("no Sensitivity rows for method $(repr(method)) at tuning criterion $(repr(tune))")
    return sort(r, :n)
end

_at(r::DataFrame, col::AbstractString, n::Integer) =
    n == 0 ? Float64(r[end, col]) : Float64(only(r[r.n .== n, col]))

# The Profile curve (bootstrap mean error over the grid) of one method.
function _profile_curve(profile::DataFrame, method::AbstractString, criterion::AbstractString)
    r = profile[(profile.method .== method) .& (profile.criterion .== criterion), :]
    isempty(r) && error("no Profile rows for method $(repr(method)) at criterion $(repr(criterion))")
    return Vector{Float64}(sort(r, :grid_idx).error_mean)
end

"""
    prose_number(cfg, entry) -> Vector{Float64}

The statistic `entry` of configs/paper_numbers.toml describes, unrounded: one
value, or a `[lo, hi]` pair for the range kinds.
"""
function prose_number(cfg::AbstractDict, e::AbstractDict)
    srcs = haskey(e, "sources") ? e["sources"] : [e["source"]]
    runs = srcs                       # the [inputs.<id>] ids of the stored runs
    kind = e["kind"]
    if kind in ("span_max", "span_range")
        spans = Float64[]
        for run in runs, m in e["methods"]
            y = _profile_curve(JLD2.load(input_path(cfg, run), "profile"), m, cfg["check"]["criterion"])
            push!(spans, maximum(y) - minimum(y))
        end
        return kind == "span_max" ? [maximum(spans)] : [minimum(spans), maximum(spans)]
    end
    col, n, tune = e["column"], get(e, "n", 0), cfg["check"]["tune"]
    if kind == "convergence_ratio"
        # One run. A method's convergence size: the smallest validation size n
        # at which `column` is within the relative `tolerance` of its own
        # asymptote (its value at the largest n). The ratio: the smallest
        # convergence size of `methods` over the largest of `against` — how
        # many times more validation data EVERY method of the first group needs
        # than EVERY method of the second.
        length(runs) == 1 || error("$(e["id"]): kind convergence_ratio takes one source")
        s = JLD2.load(input_path(cfg, only(runs)), "summary")
        function conv_n(m)
            r = _sens_rows(s, m, tune)
            y = Vector{Float64}(r[!, col])
            return r.n[findfirst(<=(y[end] * (1 + e["tolerance"])), y)]
        end
        return [minimum(conv_n(m) for m in e["methods"]) / maximum(conv_n(m) for m in e["against"])]
    end
    if kind == "min_ratio"
        # Per run: the smallest value of `methods` over the largest of
        # `against` — the factor by which EVERY method of the first group
        # exceeds EVERY method of the second in that run. Then [min, max] over
        # the runs.
        ratios = Float64[]
        for run in runs
            s = JLD2.load(input_path(cfg, run), "summary")
            num = minimum(_at(_sens_rows(s, m, tune), col, n) for m in e["methods"])
            den = maximum(_at(_sens_rows(s, m, tune), col, n) for m in e["against"])
            push!(ratios, num / den)
        end
        return [minimum(ratios), maximum(ratios)]
    end
    vals = Float64[]
    for run in runs
        s = JLD2.load(input_path(cfg, run), "summary")
        for m in e["methods"]
            r = _sens_rows(s, m, tune)
            push!(vals, kind == "max_gap" ? _at(r, col, n) - _at(r, col, 0) : _at(r, col, n))
        end
    end
    kind == "value" && length(vals) != 1 &&
        error("$(e["id"]): kind value takes one method in one run")
    kind == "value" && return vals
    kind in ("max", "max_gap") && return [maximum(vals)]
    kind == "min" && return [minimum(vals)]
    kind == "range" && return [minimum(vals), maximum(vals)]
    error("$(e["id"]): unknown kind $(repr(kind))")
end

"""
    check_prose_number(cfg, entry) -> NamedTuple

The statistic, rounded to the entry's `digits`, against its `stated` value:
`relation = "eq"` — the rounded value(s) equal the stated one(s); `"lt"` — the
value lies below the stated bound.
"""
function check_prose_number(cfg::AbstractDict, e::AbstractDict)
    v = prose_number(cfg, e)
    stated = Float64.(e["stated"] isa AbstractVector ? e["stated"] : [e["stated"]])
    rounded = round.(v; digits = e["digits"])
    ok = e["relation"] == "lt" ? all(v .< stated) :
         e["relation"] == "eq" ? rounded == stated : error("$(e["id"]): relation must be eq or lt")
    return (id = String(e["id"]), computed = v, rounded = rounded, stated = stated,
            relation = String(e["relation"]), ok = ok)
end

"""
    prose_numbers(cfg, rows; run_key, commit) -> PaperNumbers

The numbers the manuscript reads as macros rather than stating them as
literals, as VisualGeometryToolkitEval's number macros (`PaperNumbers`): one
macro per name of an entry's `macros`, paired with its rounded value(s) and
typeset with exactly `digits` decimals (`add_number!`). `rows` are the
`check_prose_number` results, in the order of `cfg["numbers"]`; `run_key` and
`commit` are the provenance of the run the numbers are computed in (its key and
recorded `gitcommit`). `write_numbers` writes the fragment.
"""
function prose_numbers(cfg::AbstractDict, rows; run_key::AbstractString, commit::AbstractString)
    nums = PaperNumbers(; driver = "scripts/paper_numbers.jl", config = "configs/paper_numbers.toml", run_key, commit)
    for (e, r) in zip(cfg["numbers"], rows)
        haskey(e, "macros") || continue
        length(e["macros"]) == length(r.rounded) ||
            error("$(e["id"]): $(length(e["macros"])) macro name(s) for $(length(r.rounded)) value(s)")
        r.ok || error("$(e["id"]): the computed value $(r.rounded) is not the stated $(r.stated); a macro is generated only for a number that checks")
        for (name, v) in zip(e["macros"], r.rounded)
            add_number!(nums, name, v; digits = e["digits"])
        end
    end
    return nums
end
