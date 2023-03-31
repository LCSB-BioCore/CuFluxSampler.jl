"""
$(README)
"""
module CuFluxSampler
using DocStringExtensions

import COBREXA
import CUDA
import SparseArrays

include("TeaRNG.jl")
include("FullAffineHR.jl")
include("AffineHR.jl")
include("ACHR.jl")

end # module CuFluxSampler
