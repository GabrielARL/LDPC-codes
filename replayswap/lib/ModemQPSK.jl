module ModemQPSK

export QPSK_MAP, bpsk_from_bits, qpsk_from_bits

# Convention (matches your demos/jsdc_utils.jl):
# bit = 1 -> +1
# bit = 0 -> -1
@inline _axis(bit::Int) = (bit == 1) ? 1.0 : -1.0

# QPSK_MAP indexed by (bI<<1 | bQ) + 1, where bI,bQ ∈ {0,1}
# So order is: 00, 01, 10, 11
const QPSK_MAP = ComplexF64[
    (_axis(0) + 1im*_axis(0))/sqrt(2),  # 00 -> (-1 - i)/√2
    (_axis(0) + 1im*_axis(1))/sqrt(2),  # 01 -> (-1 + i)/√2
    (_axis(1) + 1im*_axis(0))/sqrt(2),  # 10 -> (+1 - i)/√2
    (_axis(1) + 1im*_axis(1))/sqrt(2),  # 11 -> (+1 + i)/√2
]

function bpsk_from_bits(bits::AbstractVector{<:Integer})
    out = Vector{Float64}(undef, length(bits))
    @inbounds for i in eachindex(bits)
        out[i] = _axis(Int(bits[i]))
    end
    return out
end

function qpsk_from_bits(bits::AbstractVector{<:Integer})
    @assert iseven(length(bits)) "QPSK requires even-length bit vector"
    Ns = length(bits) ÷ 2
    s = Vector{ComplexF64}(undef, Ns)
    @inbounds for k in 1:Ns
        bI = Int(bits[2k-1])
        bQ = Int(bits[2k])
        idx = (bI << 1) | bQ
        s[k] = QPSK_MAP[idx+1]
    end
    return s
end

end # module

