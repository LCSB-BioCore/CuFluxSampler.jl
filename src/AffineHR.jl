module AffineHR
using ..CUDA
using ..DocStringExtensions

import ..COBREXA
import ..TeaRNG

"""
$(TYPEDSIGNATURES)

Use the affine-combination hit-and-run algorithm to generate a sample of the
feasible area of `m` from the `warmup` points supplied as columns in a matrix.
If you are generating a sample of the optimal solution, it is expected that the
optimum bound is already present in `m`.

Returns a matrix of `npts` samples organized in columns.
"""
function sample(m::COBREXA.MetabolicModel, warmup::AbstractMatrix, npts::Int, iters::Int)
    epsilon = 1e-5

    # allocate everything
    base_points = cu(Matrix{Float32}(warmup))
    ws = CUDA.zeros(size(base_points, 2), npts)
    dirs = CUDA.zeros(size(base_points, 1), npts)
    lblambdas = CUDA.zeros(size(dirs))
    ublambdas = CUDA.zeros(size(dirs))
    lmins = CUDA.zeros(size(dirs))
    lmaxs = CUDA.zeros(size(dirs))
    lmin = CUDA.zeros(1, npts)
    lmax = CUDA.zeros(1, npts)
    newpts = CUDA.zeros(size(pts))
    lws = CUDA.zeros(1, npts)
    oks = CUDA.zeros(Bool, 1, npts)

    # extract model data
    S = CUDA.CUSPARSE.CuSparseMatrixCSR(Float32.(stoichiometry(m)))
    lbsc, ubsc = bounds(m)
    lbs = cu(lbsc)
    ubs = cu(ubsc)

    # pre-generate first batch of the points
    @cuda threads = 256 blocks = 32 TeaRNG.device_fill_rand!(ws, 0)
    pts = (base_points * ws) ./ sum(ws, dims = 1)

    # run the iterations
    @time for iter = 1:iters
        # make random point combinations and convert to directions
        @cuda threads = 256 blocks = 32 TeaRNG.device_fill_rand!(ws, iter * 2)
        dirs .= ((base_points * ws) ./ sum(ws, dims = 1)) .- pts

        # unit-size directions
        dirs ./= sqrt.(sum(dirs .^ 2, dims = 1))

        # compute lower and upper bounds on lambdas
        lblambdas .= (lbs .- pts) ./ dirs
        ublambdas .= (ubs .- pts) ./ dirs
        lmins .= min.(lblambdas, ublambdas)
        lmaxs .= max.(lblambdas, ublambdas)
        lmin .= maximum(ifelse.(isfinite.(lmins), lmins, -Inf32), dims = 1)
        lmax .= minimum(ifelse.(isfinite.(lmaxs), lmaxs, Inf32), dims = 1)

        # generate random lambdas and compute new points 
        @cuda threads = 256 blocks = 32 TeaRNG.device_fill_rand!(lws, iter * 2 + 1)
        newpts .= pts + dirs .* (lmin .+ lws .* (lmax .- lmin))

        # check if the new points are okay and mask the replacement if not
        oks .= (sum((S * newpts) .^ 2, dims = 1) .< epsilon)
        oks .= oks .&& all(newpts .>= lbs .&& newpts .<= ubs, dims = 1)
        pts .= ifelse.(oks, newpts, pts)
    end

    collect(pts)
end

end
