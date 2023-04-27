module ACHR
using ..CUDA
using ..DocStringExtensions
using SparseArrays

import ..COBREXA
import ..TeaRNG
import Random

"""
$(TYPEDSIGNATURES)

A traditional artificially-centered hit-and-run algorithm that starts with
`start` points.

Refer to the documentation in module AffineHR for the meaning of arguments.
"""
function sample(
    m::COBREXA.MetabolicModel,
    start::AbstractMatrix;
    iters::Int,
    bound_stoichiometry::Bool = false,
    check_stoichiometry::Bool = true,
    direction_noise_max::Union{Nothing,Float32} = nothing,
    epsilon::Float32 = 1.0f-5,
    seed = Random.rand(UInt32),
)
    # allocate base helper variables
    npts = size(start, 2)
    pts = cu(Matrix{Float32}(start))
    dirs = CUDA.zeros(size(start, 1), npts)
    lblambdas = CUDA.zeros(size(dirs))
    ublambdas = CUDA.zeros(size(dirs))
    lmins = CUDA.zeros(size(dirs))
    lmaxs = CUDA.zeros(size(dirs))
    lmin = CUDA.zeros(1, npts)
    lmax = CUDA.zeros(1, npts)
    lws = CUDA.zeros(1, npts)
    oks = CUDA.zeros(Bool, 1, npts)

    # extract model data
    S = CUDA.CUSPARSE.CuSparseMatrixCSR(Float32.(COBREXA.stoichiometry(m)))
    lbs, ubs = cu.(COBREXA.bounds(m))
    if check_stoichiometry || bound_stoichiometry
        b = cu(collect(COBREXA.balance(m)))
    end

    # conditional parts/allocations
    if bound_stoichiometry
        btmp = CUDA.zeros(length(b), npts)
        bdirs = CUDA.zeros(size(btmp))
        blblambdas = CUDA.zeros(size(btmp))
        bublambdas = CUDA.zeros(size(btmp))
        blmins = CUDA.zeros(size(btmp))
        blmaxs = CUDA.zeros(size(btmp))
    end

    bound_coupling = COBREXA.n_coupling_constraints(m) > 0
    if bound_coupling
        C = CUDA.CUSPARSE.CuSparseMatrixCSR(Float32.(COBREXA.coupling(m)))
        clbs, cubs = cu.(COBREXA.coupling_bounds(m))
        ctmp = CUDA.zeros(size(C, 1), npts)
        cdirs = CUDA.zeros(size(ctmp))
        clblambdas = CUDA.zeros(size(ctmp))
        cublambdas = CUDA.zeros(size(ctmp))
        clmins = CUDA.zeros(size(ctmp))
        clmaxs = CUDA.zeros(size(ctmp))
    end

    add_noise = !isnothing(direction_noise_max)
    if add_noise
        noise_offset = -direction_noise_max
        noise_scale = 2.0f0 * direction_noise_max
    end

    # swap buffer for pts
    newpts = CUDA.zeros(size(pts))

    # run the iterations
    for iter = 1:iters

        dirs .= (sum(pts; dims = 2) ./ npts) .- pts

        if add_noise
            @cuda threads = 256 blocks = 32 TeaRNG.device_add_unif_rand!(
                dirs,
                UInt32(seed + UInt32(iter * 2)),
                noise_offset,
                noise_scale,
            )
        end

        # unit-size directions
        dirs ./= sqrt.(sum(dirs .^ 2, dims = 1))

        # compute lower and upper bounds on lambdas by variable bounds
        lblambdas .= (lbs .- pts) ./ dirs
        ublambdas .= (ubs .- pts) ./ dirs
        lmins .= min.(lblambdas, ublambdas)
        lmaxs .= max.(lblambdas, ublambdas)
        lmin .= maximum(ifelse.(isfinite.(lmins), lmins, -Inf32), dims = 1)
        lmax .= minimum(ifelse.(isfinite.(lmaxs), lmaxs, Inf32), dims = 1)

        if bound_stoichiometry
            btmp .= S * pts .- b
            bdirs .= S * dirs
            blblambdas .= (-epsilon .- btmp) ./ bdirs
            bublambdas .= (epsilon .- btmp) ./ bdirs
            blmins .= min.(blblambdas, bublambdas)
            blmaxs .= max.(blblambdas, bublambdas)
            lmin .=
                max.(lmin, maximum(ifelse.(isfinite.(blmins), blmins, -Inf32), dims = 1))
            lmax .= min.(lmax, minimum(ifelse.(isfinite.(blmaxs), blmaxs, Inf32), dims = 1))
        end

        if bound_coupling
            ctmp .= S * pts
            cdirs .= S * dirs
            clblambdas .= (clbs .- ctmp .- epsilon) ./ cdirs
            cublambdas .= (cubs .- ctmp .+ epsilon) ./ cdirs
            clmins .= min.(clblambdas, cublambdas)
            clmaxs .= max.(clblambdas, cublambdas)
            lmin .=
                max.(lmin, maximum(ifelse.(isfinite.(clmins), clmins, -Inf32), dims = 1))
            lmax .= min.(lmax, minimum(ifelse.(isfinite.(clmaxs), clmaxs, Inf32), dims = 1))
        end

        # generate random lambdas and compute new points 
        @cuda threads = 256 blocks = 32 TeaRNG.device_fill_rand!(
            lws,
            seed + UInt32(iter * 2 + 1),
        )
        newpts .= pts + dirs .* (lmin .+ lws .* (lmax .- lmin))

        oks .= all((newpts .>= lbs) .& (newpts .<= ubs), dims = 1)

        if check_stoichiometry
            # check if the new points balance is within the equality bounds
            oks .&= (sum((S * newpts .- b) .^ 2, dims = 1) .< epsilon)
        end

        pts .= ifelse.(oks, newpts, pts)
    end

    collect(pts)
end

end # module AffineHR
