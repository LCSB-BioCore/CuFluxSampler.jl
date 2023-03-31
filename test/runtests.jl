
using Test, CuFluxSampler

using COBREXA
using GLPK

include("data_downloaded.jl")

@testset "CuFluxSampler tests" begin
    include("FullAffHR.jl")
    include("xHR.jl")
end
