
@testset "General hit-and-run schemes" begin

    m = load_model(df("e_coli_core.xml"))
    start = warmup_from_variability(m, GLPK.Optimizer)
    start = permutedims(repeat(start', 6)[1:1024, :], (2, 1))
    lbs, ubs = bounds(m)
    epsilon = 1.0f-5

    @testset "module $(mod)" for mod in [CuFluxSampler.AffineHR, CuFluxSampler.ACHR]
        @testset "stoichiometry bounding $bound_stoichiometry" for bound_stoichiometry in
                                                                   [false, true]
            @testset "stoichiometry checks $check_stoichiometry" for check_stoichiometry in
                                                                     [false, true]
                @testset "direction noise $direction_noise_max" for direction_noise_max in
                                                                    [nothing, 1.0f-5]

                    sample = mod.sample(
                        m,
                        start;
                        iters = 10,
                        bound_stoichiometry,
                        check_stoichiometry,
                        direction_noise_max,
                        epsilon,
                    )

                    @test all(sample .>= lbs .- epsilon)
                    @test all(sample .<= ubs .+ epsilon)
                    if bound_stoichiometry || check_stoichiometry
                        @test all((stoichiometry(m) * sample) .^ 2 .< 1e-5)
                    end
                end
            end
        end
    end
end
