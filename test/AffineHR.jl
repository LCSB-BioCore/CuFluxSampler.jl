
@testset "Affine-combination hit-and-run" begin

    #m = load_model("work/COBREXA.jl/test/downloaded/e_coli_core.xml")
    m = load_model(df("e_coli_core.xml"))
    warmup = warmup_from_variability(m, GLPK.Optimizer)
    sample = CuFluxSampler.AffineHR.sample(m, warmup, size(warmup,2), 100)

    lbs, ubs = bounds(m)
    @test all(sample .>= lbs)
    @test all(sample .<= ubs)
    @test all((stoichiometry(m) * sample).^2 .< 1e-6)
end
