# =============================================================================
# src/panels.jl — the paper's Profile and Sensitivity panel renderers: the
# pgfplots `\nextgroupplot` fragments of Fig. 1 (Profile) and Fig. 2
# (Sensitivity) and their two style sheets, drawn from an evaluated table
# (the `source.jld2` of a stored run: `profile` + `profile_oracle`, or
# `summary`).
#
# PORTED from the renderers that drew the submitted figures —
# RobustVisualGeometry eval/neurips2026/plot_profile.jl
# (`_profile_records_from_df`, `_render_profile_panel_group`,
# `_render_paper_styles`) and plot_val.jl (`load_results`, `build_axis_paper`
# + `render_panel_group`, `_render_sensitivity_paper_styles`) at 67d76e2-dirty,
# committed as 6bc825d — with the same arithmetic in the same order, so every
# coordinate is bit-identical. Two departures, both of form only:
#   * the Sensitivity fragment is written directly in the form the paper
#     carries since its arXiv preparation (`coordinates {…}` instead of
#     PGFPlotsX's inline `table[row sep={\\}]`, a text conversion of the same
#     numbers), so no PGFPlotsX round trip sits between the table and the file;
#   * a Profile cell may be drawn without its confidence band (`bands = false`):
#     the submitted ETH3D fragment carries none, and the header the renderer
#     wrote without bands.
# Everything a fragment states that is not a number from the table (titles,
# file names, which cells, which criterion) comes from the driver's config of
# record; the house style (colours, marks, panel labels) is the table below.
# =============================================================================
using DataFrames: DataFrame
using Printf: Printf

# ---- house style (plot_val.jl STYLE / _PAPER_COLOR_HEX / PANELS) -----------------

const PAPER_COLOR_HEX = Dict(
    "RANSAC" => "ffb6c1", "MSAC" => "ffe119", "GaU" => "42d4f4", "MAGSAC" => "f58231",
    "Jeffreys" => "e6194b", "IG-Robust" => "3cb44b", "IG-MLE" => "4363d8", "Oracle" => "808080",
    "ransac-PerPairBest" => "ffb6c1", "msac-PerPairBest" => "ffe119",
    "gau-PerPairBest" => "42d4f4", "magsac-PerPairBest" => "f58231",
)

paper_color_name(k::AbstractString) = "Color" * replace(k, "-" => "")

# (method key in the summary, display label, colour name, mark, dashed)
const STYLE = [
    ("RANSAC", "RANSAC", paper_color_name("RANSAC"), "x", false),
    ("MSAC", "MSAC", paper_color_name("MSAC"), "diamond*", false),
    ("GaU", "GaU", paper_color_name("GaU"), "triangle*", false),
    ("MAGSAC", "MAGSAC++", paper_color_name("MAGSAC"), "star", false),
    ("Jeffreys", "Ours (Jeffreys)", paper_color_name("Jeffreys"), "pentagon", false),
    ("IG-Robust", "Ours (IG-Robust)", paper_color_name("IG-Robust"), "o", false),
    ("IG-MLE", "Ours (IG-MLE)", paper_color_name("IG-MLE"), "square", false),
    ("Oracle", "Oracle", paper_color_name("Oracle"), "asterisk", true),
    ("ransac-PerPairBest", raw"RANSAC (per-pair $\tau^*$)", paper_color_name("ransac-PerPairBest"), "x", true),
    ("msac-PerPairBest", raw"MSAC (per-pair $\tau^*$)", paper_color_name("msac-PerPairBest"), "diamond*", true),
    ("gau-PerPairBest", raw"GaU (per-pair $\tau^*$)", paper_color_name("gau-PerPairBest"), "triangle*", true),
    ("magsac-PerPairBest", raw"MAGSAC++ (per-pair $\tau^*$)", paper_color_name("magsac-PerPairBest"), "star", true),
]

# Oracle-style references the headline Sensitivity panels leave out.
const SENSITIVITY_EXCLUDE = Set(["Oracle", "ransac-PerPairBest", "msac-PerPairBest",
                                 "gau-PerPairBest", "magsac-PerPairBest"])

# Profile method tag → STYLE key, in drawing order (baselines behind, ours on top).
const PROFILE_STYLE_LOOKUP = [
    :ransac => "RANSAC", :msac => "MSAC", :gau => "GaU", :magsac => "MAGSAC",
    :jeffreys => "Jeffreys", :ig_robust => "IG-Robust", :ig_mle => "IG-MLE",
]

# Sensitivity panels: (id, columns of the summary, y label).
const SENSITIVITY_PANELS = (
    (id = "eR_mean", cols = (:mean_med, :lo95_mean_med, :hi95_mean_med), ylabel = raw"$\mathbb{E}[e_{\mathbf{R}}]$ (deg)"),
    (id = "eR_std", cols = (:std_med, :lo95_std_med, :hi95_std_med), ylabel = raw"$\mathrm{std}[e_{\mathbf{R}}]$ (deg)"),
    (id = "mAA_mean", cols = (:mean_mAA, :lo95_mean_mAA, :hi95_mean_mAA), ylabel = raw"$\mathbb{E}[\mathrm{mAA}]$"),
    (id = "mAA_std", cols = (:std_mAA, :lo95_std_mAA, :hi95_std_mAA), ylabel = raw"$\mathrm{std}[\mathrm{mAA}]$"),
)

function sensitivity_panel(id::AbstractString)
    i = findfirst(p -> p.id == id, SENSITIVITY_PANELS)
    i === nothing && error("no Sensitivity panel $(repr(id)); known: $(join((p.id for p in SENSITIVITY_PANELS), ", "))")
    return SENSITIVITY_PANELS[i]
end

_style_row(key) = (i = findfirst(r -> r[1] == key, STYLE); i === nothing ? nothing : STYLE[i])

# ---- Profile -------------------------------------------------------------------------

"""
    profile_records(profile::DataFrame, criterion) -> Vector{NamedTuple}

One record per method present for `criterion` (`"median"`), in drawing order:
the swept widths `xs`, the bootstrap mean `ys` and std `stds` of the error,
the bracket centre `x_star = √(min·max)` the x axis is normalised by, and the
curve's argmin. (plot_profile.jl `_profile_records_from_df`.)
"""
function profile_records(df::DataFrame, criterion::AbstractString)
    sub = df[df.criterion .== criterion, :]
    out = NamedTuple[]
    isempty(sub) && return out
    for (snake_key, style_key) in PROFILE_STYLE_LOOKUP
        rows = sub[sub.method .== String(snake_key), :]
        isempty(rows) && continue
        rows = sort(rows, :grid_idx)
        st = _style_row(style_key)
        st === nothing && continue
        xs = Vector{Float64}(rows.tau_or_a)
        ys = Vector{Float64}(rows.error_mean)
        stds = Vector{Float64}(rows.error_std)
        x_center = sqrt(minimum(xs) * maximum(xs))
        j = argmin(ys)
        push!(out, (snake_key = snake_key, label = st[2], color = st[3], mark = st[4], dashed = st[5],
                    xs = xs, ys = ys, stds = stds, x_star = x_center, y_star = ys[j], x_argmin = xs[j]))
    end
    return out
end

"""
    profile_references(oracle::DataFrame, criterion) -> (oracle_y, per_method_best)

The GT-Oracle line (the `"GT-Oracle"` row) and each method's per-pair-best
envelope (every other row), from a run's `profile_oracle` table.
"""
function profile_references(odf::DataFrame, criterion::AbstractString)
    gt = odf[(odf.method .== "GT-Oracle") .& (odf.criterion .== criterion), :]
    oracle_y = isempty(gt) ? NaN : Float64(first(gt.oracle_mean))
    per_method_best = Dict{Symbol,Float64}()
    for row in eachrow(odf[odf.criterion .== criterion, :])
        row.method == "GT-Oracle" && continue
        per_method_best[Symbol(row.method)] = row.oracle_mean
    end
    return oracle_y, per_method_best
end

_paper_style_name(snake_key::Symbol) = "ours_" * String(snake_key)

"""
    write_profile_group(io, records, title; oracle_y = NaN, per_method_best = Dict(), bands = true)

The Profile `\\nextgroupplot` fragment of one cell (plot_profile.jl
`_render_profile_panel_group`): per method its ±1.96σ band (unless
`bands = false`), its mean curve on x = log10(κ/κ⋆), its per-pair-best
envelope and its argmin marker, then the GT-Oracle line.
"""
function write_profile_group(io::IO, records, title::AbstractString;
                             oracle_y::Float64 = NaN, per_method_best::AbstractDict = Dict{Symbol,Float64}(),
                             bands::Bool = true)
    println(io, "% panel_profile_group.tex — \\nextgroupplot fragment.")
    if bands
        println(io, "% Host preamble MUST load: \\usepgfplotslibrary{groupplots},")
        println(io, "%                          \\usepgfplotslibrary{fillbetween}.")
    end
    println(io, "% Must be \\input inside a \\begin{groupplot}[...] with these")
    println(io, "% options set at the GROUPPLOT level (not per-cell):")
    println(io, "%   panel_axis, xmin=-1, xmax=1, xtick={\\PanelXTicks},")
    println(io, "%   xlabel={\\PanelXLabel}, ylabel={\\PanelYLabel},")
    println(io, "%   ylabels at=edge left, yticklabels at=edge left,")
    println(io, "%   group style={group size=N by 1, horizontal sep=...}")
    println(io, "% That way pgfplots' native `ylabels at=edge left` shows the y-")
    println(io, "% annotations only on the leftmost cell.  Per-cell options here")
    println(io, "% are LIMITED to `title=` so the row's wrapper retains all axis")
    println(io, "% control.  See `profile_row_template` for the canonical wrapper.")
    println(io)
    println(io, "\\nextgroupplot[title={", title, "}]")
    for r in records
        xn = @. log10(r.xs / r.x_star)
        sn = _paper_style_name(r.snake_key)
        if bands
            upper = r.ys .+ 1.96 .* r.stds
            lower = r.ys .- 1.96 .* r.stds
            u_name = "U_" * String(r.snake_key) * "_grp"
            l_name = "L_" * String(r.snake_key) * "_grp"
            print(io, "  \\addplot[name path=", u_name, ", draw=none, mark=none, forget plot] coordinates {")
            for (xi, yi) in zip(xn, upper); print(io, "(", xi, ",", yi, ") "); end
            println(io, "};")
            print(io, "  \\addplot[name path=", l_name, ", draw=none, mark=none, forget plot] coordinates {")
            for (xi, yi) in zip(xn, lower); print(io, "(", xi, ",", yi, ") "); end
            println(io, "};")
            println(io, "  \\addplot[", sn, ", fill, fill opacity=0.18, draw=none, mark=none, forget plot] ",
                    "fill between [of=", u_name, " and ", l_name, "];")
        end
        print(io, "  \\addplot+[", sn, ", mark=none] coordinates {")
        for (xi, yi) in zip(xn, r.ys); print(io, "(", xi, ",", yi, ") "); end
        println(io, "};")
        pmb = get(per_method_best, r.snake_key, NaN)
        if isfinite(pmb)
            println(io, "  \\addplot+[", sn, ", mark=none, line width=0.4pt, dashed, forget plot] ",
                    "coordinates {(-1.0,", pmb, ") (1.0,", pmb, ")};")
        end
        argmin_xn = log10(r.x_argmin / r.x_star)
        println(io, "  \\addplot+[", sn, ", mark=*, only marks, mark size=\\PanelArgminSize, ",
                "forget plot] coordinates {(", argmin_xn, ",", r.y_star, ")};")
    end
    if isfinite(oracle_y)
        println(io, "  \\addplot+[ours_oracle, mark=none, forget plot] ",
                "coordinates {(-1.0,", oracle_y, ") (1.0,", oracle_y, ")};")
    end
    return nothing
end

"""
    write_profile_styles(styles_io, shared_io, records, criterion)

The Profile style sheets (plot_profile.jl `_render_paper_styles`): the shared
palette and per-method `ours_<m>` / `ours_<m>_marker` styles, and the
cell-specific `\\Panel*` defaults with the `panel_axis` style.
"""
function write_profile_styles(styles_io::IO, shared_io::IO, records, criterion::AbstractString)
    ylab = criterion == "median" ? raw"median $e_{\mathbf{R}}$ (deg)" : raw"$-\mathrm{mAA}@10^\circ$"
    io = shared_io
    println(io, "% paper_styles_shared.tex — colour palette + per-method named")
    println(io, "% styles for the paper-mode panels.  Identical across all")
    println(io, "% cells/experiments.  \\input once into your paper preamble")
    println(io, "% (or copy verbatim).  Standalone preview.tex \\input's this")
    println(io, "% before any panel; main.tex should do the same once globally.")
    println(io)
    println(io, "% --- Colour palette (\\providecolor: paper-side override wins) ---")
    for (k, hex) in sort(collect(PAPER_COLOR_HEX); by = first)
        println(io, "\\providecolor{", paper_color_name(k), "}{HTML}{", uppercase(hex), "}")
    end
    println(io)
    println(io, "% --- Per-method addplot styles ---")
    for r in records
        sn = _paper_style_name(r.snake_key)
        line_style = r.dashed ? "dashed" : "solid"
        println(io, "\\pgfplotsset{", sn, "/.style={color=", r.color, ", line width=1.0pt, ",
                line_style, ", mark options={solid}}}")
        println(io, "\\pgfplotsset{", sn, "_marker/.style={", sn, ", mark=", r.mark, ", mark size=2pt, ",
                "mark options={solid}}}")
    end
    println(io, "\\pgfplotsset{ours_oracle/.style={color=black, line width=1.0pt, solid}}")

    io = styles_io
    println(io, "% paper_styles.tex — cell-/criterion-specific bits.")
    println(io, "% Standalone preview: \\input{paper_styles_shared}\\input{paper_styles}")
    println(io, "% BEFORE \\input{panel_profile_paper}.  In main.tex,")
    println(io, "% \\input{paper_styles_shared} once globally, omit this file,")
    println(io, "% and \\renewcommand the \\Panel* macros per panel-input as needed.")
    println(io)
    println(io, "\\providecommand{\\PanelXLabel}{\$\\kappa / \\kappa^\\star\$ \\ \\ (factor relative to method-specific optimum)}")
    println(io, "\\providecommand{\\PanelYLabel}{", ylab, "}")
    println(io, "\\providecommand{\\PanelXTicks}{-1,-0.5,0,0.5,1}")
    println(io, "\\providecommand{\\PanelXTickLabels}{\$\\tfrac{1}{10}\$,\$\\tfrac{1}{\\sqrt{10}}\$,1,\$\\sqrt{10}\$,10}")
    println(io, "\\providecommand{\\PanelArgminSize}{1.5pt}")
    println(io)
    _panel_axis_style(io)
    return nothing
end

function _panel_axis_style(io::IO)
    println(io, "% Named axis style — panels opt-in via `panel_axis` in their")
    println(io, "% axis options.  Legends/insets/color-bars stay clean.")
    println(io, "% NOTE: visual properties only — NO width/height.  Layout (size,")
    println(io, "% subfigure dimensions, groupplot geometry) is the paper's call;")
    println(io, "% baking width/height here would silently override paper-side")
    println(io, "% sizing via `every axis/.append style{}` last-wins.  Standalone")
    println(io, "% preview supplies the default size in `preview.tex`.")
    println(io, "\\pgfplotsset{panel_axis/.style={")
    println(io, "    grid style={gray!20},")
    println(io, "    label style={font=\\footnotesize},")
    println(io, "    tick label style={font=\\scriptsize},")
    println(io, "}}")
    return nothing
end

# ---- Sensitivity ---------------------------------------------------------------------

"""
    sensitivity_data(summary::DataFrame) -> Dict{(method_key, tune_key) => DataFrame}

The Sensitivity summary bucketed per (method, tuning criterion), each sorted by
the validation-set size `n` (plot_val.jl `load_results`): the method key drops
a trailing `(median)`/`(mAA)`.
"""
function sensitivity_data(summary::DataFrame)
    df = copy(summary)
    df.method_key = [replace(strip(String(m)), r"\s*\((?:median|mAA)\)$"i => "") for m in df.method]
    df.tune_key = String.(df.tuning_criterion)
    df.nn = Float64.(df.n)
    sort!(df, :nn)
    data = Dict{Tuple{String,String},DataFrame}()
    for key in unique(zip(df.method_key, df.tune_key))
        data[(String(key[1]), String(key[2]))] = df[(df.method_key .== key[1]) .& (df.tune_key .== key[2]), :]
    end
    return data
end

function _sensitivity_style_name(k::AbstractString)
    endswith(k, "-PerPairBest") && return "ours_" * lowercase(replace(k, "-PerPairBest" => "")) * "_pairbest"
    return "ours_" * lowercase(replace(k, "-" => "_"))
end

_coordinates(io, xs, ys) = (print(io, "        coordinates {");
                            print(io, join(("(" * string(x) * "," * string(y) * ")" for (x, y) in zip(xs, ys)), " "));
                            println(io, "}"); println(io, "        ;"))

"""
    write_sensitivity_group(io, panel, data, tune, title)

The Sensitivity `\\nextgroupplot` fragment of one cell and one panel
(plot_val.jl `build_axis_paper` → `render_panel_group`, in the paper's
`coordinates` form): per headline method its 95 % band (`fill between` the
upper and lower CI paths) and its marked curve over the validation-set size.
"""
function write_sensitivity_group(io::IO, panel, data, tune::AbstractString, title::AbstractString)
    println(io, "% panel ..._group.tex — \\nextgroupplot fragment.")
    println(io, "% Host preamble MUST load: \\usepgfplotslibrary{groupplots},")
    println(io, "%                          \\usepgfplotslibrary{fillbetween}.")
    println(io, "% Wrap N such fragments inside one \\begin{groupplot}[...] that")
    println(io, "% sets `panel_axis, xmode=log, log basis x=2, xmajorgrids,")
    println(io, "% ymajorgrids, xlabel={…}, ylabel={…}, xtick={…},")
    println(io, "% ytick distance=…, yminorgrids, scaled y ticks=false,")
    println(io, "% yticklabel style={…}, group style={group size=N by M, …}].")
    println(io)
    println(io, "\\nextgroupplot[title={", title, "}]")
    for (idx, (k, _, _, _, dashed)) in enumerate(STYLE)
        k in SENSITIVITY_EXCLUDE && continue
        haskey(data, (k, tune)) || continue
        s = data[(k, tune)]
        y, ylo, yhi = (s[!, c] for c in panel.cols)
        keep = [isfinite(yi) for yi in y]
        any(keep) || continue
        nn_f = s.nn[keep]
        y_f = y[keep]
        ylo_f = [isfinite(ylo[i]) ? ylo[i] : y[i] for i in eachindex(y) if keep[i]]
        yhi_f = [isfinite(yhi[i]) ? yhi[i] : y[i] for i in eachindex(y) if keep[i]]
        sn = _sensitivity_style_name(k)
        upper_name = "U$(panel.id)_$(tune)_$(idx)"
        lower_name = "L$(panel.id)_$(tune)_$(idx)"
        println(io, "    \\addplot[name path={", upper_name, "}, draw={none}, mark={none}, forget plot]")
        _coordinates(io, nn_f, yhi_f)
        println(io, "    \\addplot[name path={", lower_name, "}, draw={none}, mark={none}, forget plot]")
        _coordinates(io, nn_f, ylo_f)
        println(io, "    \\addplot[fill opacity={0.18}, draw={none}, mark={none}, forget plot, ", sn, ", fill]")
        println(io, "        fill between [of=", upper_name, " and ", lower_name, "]")
        println(io, "        ;")
        println(io, "    \\addplot[", sn, "_marker", dashed ? ", dashed" : "", "]")
        _coordinates(io, nn_f, y_f)
    end
    return nothing
end

"""
    write_sensitivity_styles(styles_io, shared_io, data)

The Sensitivity style sheets (plot_val.jl `_render_sensitivity_paper_styles`):
the shared palette and the line / marked-line style of every headline method
present, and the `\\Panel*` defaults (x ticks: every validation size present).
"""
function write_sensitivity_styles(styles_io::IO, shared_io::IO, data)
    present = Set(first(k) for k in keys(data))
    io = shared_io
    println(io, "% paper_styles_shared.tex — colour palette + Sensitivity per-method")
    println(io, "% named styles.  Identical across cells/experiments; \\input once")
    println(io, "% into your paper preamble (or copy verbatim).  Override per-method")
    println(io, "% via `\\pgfplotsset{ours_<m>/.style={...}}` in your preamble.")
    println(io)
    println(io, "% --- Colour palette (\\providecolor: paper-side override wins) ---")
    for (k, hex) in sort(collect(PAPER_COLOR_HEX); by = first)
        println(io, "\\providecolor{", paper_color_name(k), "}{HTML}{", uppercase(hex), "}")
    end
    println(io)
    println(io, "% --- Per-method addplot styles ---")
    for (k, _, color, mark, dashed) in STYLE
        k in SENSITIVITY_EXCLUDE && continue
        k in present || continue
        sn = _sensitivity_style_name(k)
        print(io, "\\pgfplotsset{", sn, "/.style={", "color=", color, ", line width=0.8pt")
        dashed && print(io, ", dashed")
        println(io, "}}")
        println(io, "\\pgfplotsset{", sn, "_marker/.style={", sn, ", mark=", mark, ", mark size=2pt, mark options={solid}}}")
    end
    nn = sort!(unique(Int[n for (_, s) in data for n in s.nn]))
    io = styles_io
    println(io, "% paper_styles.tex — cell-specific defaults: \\Panel* macros + axis cosmetics.")
    println(io, "% Standalone preview: \\input{paper_styles_shared}\\input{paper_styles}")
    println(io, "% BEFORE the panel \\input.  In main.tex, \\input{paper_styles_shared}")
    println(io, "% globally, omit this file, and \\renewcommand the \\Panel* macros")
    println(io, "% per panel-input as needed (use `{}` to suppress a label).")
    println(io)
    println(io, "\\providecommand{\\PanelXLabel}{Validation set size}")
    for p in SENSITIVITY_PANELS
        println(io, "\\providecommand{\\PanelYLabel", replace(p.id, "_" => ""), "}{", p.ylabel, "}")
    end
    println(io, "\\providecommand{\\PanelXTicks}{", join(nn, ","), "}")
    println(io)
    _panel_axis_style(io)
    return nothing
end

# ---- comparison ----------------------------------------------------------------------

"""
    fragment_coordinates(text) -> Vector{Vector{NTuple{2,Float64}}}

Every `coordinates {…}` block of a pgfplots fragment, in order, parsed to
numbers: the quantitative content of a panel, for comparing a regenerated
fragment with the submitted one independently of its formatting.
"""
function fragment_coordinates(text::AbstractString)
    blocks = Vector{NTuple{2,Float64}}[]
    for m in eachmatch(r"coordinates\s*\{([^}]*)\}", text)
        pts = NTuple{2,Float64}[]
        for p in eachmatch(r"\(\s*([^,()\s]+)\s*,\s*([^,()\s]+)\s*\)", m.captures[1])
            push!(pts, (parse(Float64, p.captures[1]), parse(Float64, p.captures[2])))
        end
        push!(blocks, pts)
    end
    return blocks
end

"""
    compare_fragments(new, old) -> NamedTuple

How a regenerated fragment `new` differs from the submitted `old` (texts):
`identical` (byte for byte), `same_numbers` (every coordinate block equal,
in order), the block counts and the largest absolute coordinate difference
over blocks of equal length (`Inf` when the structure differs).
"""
function compare_fragments(new::AbstractString, old::AbstractString)
    a, b = fragment_coordinates(new), fragment_coordinates(old)
    same_shape = length(a) == length(b) && all(length.(a) .== length.(b))
    maxdiff = same_shape ? maximum((abs(p[i] - q[i]) for (x, y) in zip(a, b) for (p, q) in zip(x, y) for i in 1:2); init = 0.0) : Inf
    return (identical = new == old, same_numbers = same_shape && a == b,
            blocks_new = length(a), blocks_old = length(b), max_abs_diff = maxdiff)
end
