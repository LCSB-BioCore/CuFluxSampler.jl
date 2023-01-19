module TeaRng
using ..CUDA

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
tea_random(x::Int, y::Int) = tea_random(UInt32(x), UInt32(y))

function device_fill_rand!(arr, seed)
    index = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(arr)
        arr[i] = Float32(tea_random(seed, i)) / 0x100000000
    end
    return
end

end # module TeaRng
