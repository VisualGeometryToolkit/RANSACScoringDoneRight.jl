# The manuscript syncs with its Overleaf project (scripts/paper_sync.sh): the
# data policy (nothing of the data can travel in a push) and the sync script.
using Test
using DrWatson

@testset "paper subtree" begin
    paper = projectdir("paper")
    @testset "no data travels in a push" begin
        banned = [r"(^|/)data/", r"(^|/)results/", r"\.jld2$"i]
        hits = [relpath(joinpath(root, f), paper) for (root, _, files) in walkdir(paper) for f in files
                if any(re -> occursin(re, relpath(joinpath(root, f), paper)), banned)]
        foreach(h -> @info("data-policy hit under paper/: " * h), hits)
        @test isempty(hits)
    end
    @testset "paper_sync.sh status" begin
        script = projectdir("scripts", "paper_sync.sh")
        @test isfile(script)
        # never a credential prompt: status reports an unreachable remote and succeeds
        cmd = addenv(`$script status`, "GIT_TERMINAL_PROMPT" => "0", "GIT_ASKPASS" => "true")
        @test success(pipeline(cmd; stdout = devnull, stderr = devnull))
    end
end
