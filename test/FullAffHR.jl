
@testset "Full-affine-combination hit-and-run" begin

    m = load_model(df("e_coli_core.xml"))
    warmup = warmup_from_variability(m, GLPK.Optimizer)

    sample = CuFluxSampler.AffineHR.sample(
        m,
        warmup,
        iters = 100,
        bound_stoichiometry = true,
        check_stoichiometry = true,
        direction_noise_max = 1.0f-5,
        epsilon = 1.0f-5,
    )

    lbs, ubs = bounds(m)
    @test all(sample .>= lbs .- 1.0f-5)
    @test all(sample .<= ubs .+ 1.0f-5)
    @test all((stoichiometry(m) * sample) .^ 2 .< 1e-5)
end
