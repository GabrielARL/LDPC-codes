# lib/TEQScanUtils_BPSK.jl
#
# Optional: fast D-only scan for BPSK using pilot-only scoring.
# (No rot/conj; BPSK is real-axis.)
#
module TEQScanUtils_BPSK

using Statistics

export shift_left,
       choose_pilots_bits,
       pilot_score_from_xhat_bpsk,
       scan_best_D_fast_bpsk

@inline function shift_left(x::Vector{ComplexF64}, D::Int)
    T = length(x)
    if D <= 0
        return x
    elseif D >= T
        return fill(0.0 + 0im, T)
    else
        return vcat(x[D+1:end], fill(0.0 + 0im, D))
    end
end

function choose_pilots_bits(n::Int; frac::Float64)
    frac <= 0 && return Int[]
    Np = max(1, round(Int, frac*n))
    posf = collect(range(1, stop=n, length=Np))
    pos = unique!(clamp.(round.(Int, posf), 1, n))
    sort!(pos)
    return pos
end

# LLR convention: log P0/P1, bit=1 iff LLR<0
# With BPSK mapping bit1->+1, bit0->-1 => LLR ≈ -c*Re(x), c>0
@inline function pilot_score_from_xhat_bpsk(x::Vector{ComplexF64},
                                           b128::Vector{Int},
                                           pilot_bits::Vector{Int},
                                           σ2::Float64)
    isempty(pilot_bits) && return -Inf
    c = 2.0 / max(σ2, 1e-12)
    s = 0.0
    @inbounds for p in pilot_bits
        llr = -c * real(x[p])
        s += (1 - 2*b128[p]) * llr
    end
    return s / length(pilot_bits)
end

function scan_best_D_fast_bpsk(xhat0::Vector{ComplexF64},
                               b128::Vector{Int};
                               pilot_frac::Float64=0.10,
                               σ2::Float64=0.5,
                               Dmax::Int=25)
    pilot_bits = choose_pilots_bits(128; frac=pilot_frac)

    bestScore = -Inf
    bestD = 0
    for D in 0:Dmax
        xD = shift_left(xhat0, D)
        sc = pilot_score_from_xhat_bpsk(xD, b128, pilot_bits, σ2)
        if sc > bestScore
            bestScore = sc
            bestD = D
        end
    end
    return bestScore, bestD, pilot_bits
end

end # module
