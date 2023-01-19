"""
Fast stateless random number generator for GPUs based on TEA cipher.
"""
module TeaRNG
using ..CUDA
using ..DocStringExtensions

"""
$(TYPEDSIGNATURES)

Use TEA cipher algorithm to reproducibly generate a `seq`-th random number from
the `stream`-th random stream.
"""
function tea_random(stream::UInt32, seq::UInt32)
    v1 = stream
    v2 = seq
    s = 0x9E3779B9
    for i = 1:8
        v1 = UInt32(
            v1 + xor(
                UInt32(UInt32(v2 << 4) + 0xA341316C),
                v2 + s,
                UInt32((v2 >> 5) + 0xC8013EA4),
            ),
        )
        v2 = UInt32(
            v2 + xor(
                UInt32(UInt32(v1 << 4) + 0xAD90777D),
                v1 + s,
                UInt32((v1 >> 5) + 0x7E95761E),
            ),
        )
        s = UInt32(s + 0x9E3779B9)
    end
    return v1
end

"""
$(TYPEDSIGNATURES)

`Int`-typed overload of [`tea_random`](@ref) provided for convenience.
"""
tea_random(x::Int, y::Int) = tea_random(UInt32(x), UInt32(y))

"""
$(TYPEDSIGNATURES)

CUDA.jl grid-stride kernel that fills the array with random numbers generated
by [`tea_random`](@ref). `seed` is used as the `stream` ID, global thread index
in grid is used as the `seq`uence number.
"""
function device_fill_rand!(arr, seed)
    index = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(arr)
        arr[i] = Float32(tea_random(seed, i)) / 0x100000000
    end
    return
end

end # module TeaRng
