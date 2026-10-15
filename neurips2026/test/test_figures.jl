# =============================================================================
# test_figures.jl — the ported renderers (src/panels.jl) reproduce every
# submitted figure fragment of paper/figures byte for byte from the stored
# runs of record (configs/fig_profile.toml, fig_sensitivity.toml), and the
# prose numbers of Sec. 4 check against their data (configs/paper_numbers.toml)
# exactly, the generated ones in paper/figures/paper_numbers.tex too. Skipped, loudly, where the stored
# runs are not bound (fetch the neurips2026_paper_runs artifact:
# scripts/fetch_data.jl).
# =============================================================================
using Test
using DrWatson
include(srcdir("NeurIPS2026.jl"))

_cfg(name) = TOML.parsefile(projectdir("configs", name * ".toml"))
_submitted(name) = read(joinpath(paper_figdir(), name), String)
_render(f) = (io = IOBuffer(); f(io); String(take!(io)))

const _STORED_BOUND = try
    isdir(stored_dir(_cfg("fig_profile")))
catch
    false
end

@testset "figures and numbers from the stored runs" begin
    if !_STORED_BOUND
        @warn "stored runs not bound — figure and number checks SKIPPED (scripts/fetch_data.jl)"
        @test_skip false
    else
        @testset "Profile (Fig. 1)" begin
            cfg = _cfg("fig_profile")
            crit = cfg["profile"]["criterion"]
            for c in cfg["profile"]["cells"]
                recs = profile_records(stored_table(cfg, c["run"], "profile"), crit)
                oy, pmb = profile_references(stored_table(cfg, c["run"], "profile_oracle"), crit)
                txt = _render(io -> write_profile_group(io, recs, c["title"]; oracle_y = oy, per_method_best = pmb, bands = c["bands"]))
                @test txt == _submitted(c["asset"])
                if c["key"] == cfg["profile"]["styles_from"]
                    s, sh = IOBuffer(), IOBuffer()
                    write_profile_styles(s, sh, recs, crit)
                    @test String(take!(s)) == _submitted(cfg["render"]["styles"])
                    @test String(take!(sh)) == _submitted(cfg["render"]["styles_shared"])
                end
            end
        end
        @testset "Sensitivity (Fig. 2)" begin
            cfg = _cfg("fig_sensitivity")
            sens = cfg["sensitivity"]
            for c in sens["cells"]
                data = sensitivity_data(stored_table(cfg, c["run"], "summary"))
                for p in sens["panels"]
                    txt = _render(io -> write_sensitivity_group(io, sensitivity_panel(p), data, sens["tune"], c["title"]))
                    @test txt == _submitted(c["assets"][p])
                end
                if c["key"] == sens["styles_from"]
                    s, sh = IOBuffer(), IOBuffer()
                    write_sensitivity_styles(s, sh, data)
                    @test String(take!(s)) == _submitted(cfg["render"]["styles"])
                    @test String(take!(sh)) == _submitted(cfg["render"]["styles_shared"])
                end
            end
        end
        @testset "prose numbers (Sec. 4)" begin
            cfg = _cfg("paper_numbers")
            rows = [check_prose_number(cfg, e) for e in cfg["numbers"]]
            for r in rows
                @test r.ok || (@info "prose number differs from its data" r; false)
            end
            @test prose_macros(cfg, rows) == _submitted(cfg["render"]["macros"])
            # the generated macros, each exactly its rounded statistic
            deployed = Dict(m[1] => parse(Float64, m[2]) for m in
                            eachmatch(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", _submitted(cfg["render"]["macros"])))
            for (e, r) in zip(cfg["numbers"], rows), (name, v) in zip(get(e, "macros", String[]), r.rounded)
                @test deployed[name] == v
            end
            gap = only(r for r in rows if r.id == "std_gap_v2")
            @test gap.rounded == [2.0, 29.0] && deployed["StdGapVtwoLo"] == 2 && deployed["StdGapVtwoHi"] == 29
            # main.tex names the datasets at the two ends: ETH3D the low, PT the high
            e = only(x for x in cfg["numbers"] if x["id"] == "std_gap_v2")
            per(src) = first(prose_number(cfg, merge(e, Dict("sources" => [src]))))
            @test per("eth3d_sensitivity") < per("pt_sensitivity")
            @test round(per("eth3d_sensitivity")) == 2 && round(per("pt_sensitivity")) == 29
            # the convergence ratios main.tex states (1 % on PT, 5 % on ETH3D)
            @test deployed["ConvRatioPT"] == 128 && deployed["ConvRatioETH"] == 8
            eth = only(x for x in cfg["numbers"] if x["id"] == "eth3d_convergence_ratio")
            # why ETH3D is stated at 5 %: at the paper's 1 % it shows no advantage
            @test only(prose_number(cfg, merge(eth, Dict("tolerance" => 0.01)))) == 1.0
        end
    end
    @testset "fragment comparison" begin
        a = "\\addplot coordinates {(1.0,2.0) (3.0,4.5)};"
        @test compare_fragments(a, a).identical
        @test compare_fragments(replace(a, " (" => "  ("), a).same_numbers
        c = compare_fragments(replace(a, "4.5" => "4.25"), a)
        @test !c.same_numbers && c.max_abs_diff == 0.25
    end
end
