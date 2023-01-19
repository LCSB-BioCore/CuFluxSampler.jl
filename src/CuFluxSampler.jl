"""
$(README)
"""
module CuFluxSampler
using DocStringExtensions

import COBREXA
import CUDA
import SparseArrays

include("TeaRNG.jl")
include("AffineHR.jl")

end # module CuFluxSampler
