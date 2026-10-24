# The Stage-6 acceptance gate for the Track-B GPU port: the device path is
# bit-identical to a plain host loop on a CPU() backend, Float64, Jeffreys and a
# proper IG prior, all three redescending families. The gate itself lives in
# gate_marg_mix_ka.jl, which is not a `*_tests.jl` file (never discovered) so it
# still runs standalone — `julia --project=. test/gate_marg_mix_ka.jl [--cuda]`
# adds a CUDABackend run from an environment that provides CUDA.
@testitem "marg_mix_gate" begin
    include(joinpath(pkgdir(RANSACScoringDoneRight), "test", "gate_marg_mix_ka.jl"))
    @testset "Track-B mixture + full-data marginal bit-exact gate" begin
        @test run_gate() == 0
    end
end
