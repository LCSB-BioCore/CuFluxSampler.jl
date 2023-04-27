module AffineHR
using ..CUDA
using ..DocStringExtensions
using SparseArrays

import ..COBREXA
import ..TeaRNG
import Random

function random_mix_matrix(npts, mix_points)
    mtx = sparse(
        Random.rand(1:npts, npts * mix_points),
        repeat(1:npts, inner = mix_points),
        Random.rand(npts * mix_points),
        npts,
        npts,
    )
    mtx ./ sum(mtx, dims = 1)
end

function random_permute_matrix(npts)
    sparse(1:npts, Random.randperm(npts), 1.0, npts, npts)
end

"""
$(TYPEDSIGNATURES)

Use the affine-combination hit-and-run algorithm to generate a sample of the
feasible area of `m` from the set of `start` points supplied as columns in a
matrix.

The run directions are generated from random affine combination of `mix_points`
points (by default 3); matrices `mix_mtx` and `permute_mtx` give fine control
about the mixing in the process. Preferably, this matrix is very sparse.

`check_stoichiometry` allows to turn on/off the filtering of generated points
based on whether they are close to the steady state (with tolerance `epsilon`).
`bound_stoichiometry` additionally computes run bounds based on the
steady-state region, and uses it to generate better runs. This is useful in
combination with `direction_noise_max` which may add a small noise to the
generated run directions, allowing the sampler to discover new directions
(potentially not obvious from warmup in `start`), but easily explodes without
limiting the directions.

Additional bounds on run ranges are taken from model coupling constraints, if
present.

If you are generating a sample of the optimal model solution, it is expected
that the optimum bound is already present in `m`.

Returns a matrix of the same size as `start`.
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
    mix_points = 3,
    mix_mtx = random_mix_matrix(size(start, 2), mix_points),
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
    mix = CUDA.CUSPARSE.CuSparseMatrixCSR(Float32.(mix_mtx))

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

        dirs .= (pts * mix) .- pts

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

        newpts .= ifelse.(oks, newpts, pts)
        pts .= newpts * mix
    end

    collect(pts)
end

end # module AffineHR
