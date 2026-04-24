# lib/TEQScanUtils.jl
#
# Utilities for TurboEQ alignment scan (D,rot,conj):
#   - FAST: uses pilot-only scoring (no full LLR vectors)
#   - CONSISTENT with ModemQPSK mapping:
#       bit=1 -> +1 axis, bit=0 -> -1 axis
#     and LLR convention:
#       LLR = log P(bit=0)/P(bit=1)
#       => bit=1 iff LLR < 0
#
# So for a soft symbol x:
#   LLR_I ≈ -c*Re(x),  LLR_Q ≈ -c*Im(x)

module TEQScanUtils

using LinearAlgebra, Statistics, Random

# Quadrant rotations
const ROT_SET = ComplexF64[1+0im, 0+1im, -1+0im, 0-1im]

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

@inline function apply_rot_conj(x::Vector{ComplexF64}, rot::ComplexF64, cj::Bool)
    cj ? conj.(x .* rot) : (x .* rot)
end

# Apply candidate to (y,h) consistently.
# NOTE: shift applies to y only (timing), rot/conj applies to both.
@inline function apply_candidate(y::Vector{ComplexF64}, h::Vector{ComplexF64}, D::Int, rot::ComplexF64, cj::Bool)
    yD = shift_left(y, D)
    if cj
        yC = conj.(yD .* rot)
        hC = conj.(h  .* rot)
    else
        yC = yD .* rot
        hC = h  .* rot
    end
    return yC, hC
end

# Evenly spaced pilot positions in 1..n (bit positions)
function choose_pilots_bits(n::Int; frac::Float64)
    frac <= 0 && return Int[]
    Np = max(1, round(Int, frac*n))
    posf = collect(range(1, stop=n, length=Np))
    return unique!(clamp.(round.(Int, posf), 1, n))
end

# Convert bit-pilot positions to symbol-pilot positions for QPSK (2 bits/sym).
function bits_to_syms_pilots(bitpos::Vector{Int})
    isempty(bitpos) && return Int[]
    sympos = Vector{Int}(undef, length(bitpos))
    @inbounds for i in eachindex(bitpos)
        sympos[i] = (bitpos[i] + 1) >>> 1
    end
    sort!(unique!(sympos))
    return sympos
end

# ------------------------------------------------------------
# FAST pilot score (no allocations)
#
# LLR convention: log P0/P1, bit=1 iff LLR<0
# With ModemQPSK mapping: bit=1 -> +1 axis, bit=0 -> -1 axis
# So LLR_I ≈ -c*Re(x), LLR_Q ≈ -c*Im(x) for any c>0
# ------------------------------------------------------------
@inline function pilot_score_from_xhat(x::Vector{ComplexF64},
                                      b1024::Vector{Int},
                                      pilot_bits::Vector{Int},
                                      σ2::Float64)
    isempty(pilot_bits) && return -Inf
    c = 2.0 / max(σ2, 1e-12)   # any positive monotone scale works for ranking
    s = 0.0
    @inbounds for p in pilot_bits
        sym = (p + 1) >>> 1
        # odd bit -> I, even bit -> Q
        llr = isodd(p) ? (-c * real(x[sym])) : (-c * imag(x[sym]))
        s += (1 - 2*b1024[p]) * llr
    end
    return s / length(pilot_bits)
end

# ------------------------------------------------------------
# Main scan: xhat0 -> best (D,rot,conj)
# ------------------------------------------------------------
"""
    scan_best_alignment_fast(xhat0, b1024; pilot_frac=0.10, σ2=0.5, Dmax=25, scan_max_bits=2000)

Returns: (bestScore, bestD, bestRot, bestConj, pilot_bits_used)

- Uses pilot-only score derived from xhat0 (no joint BCJR).
- If scan_max_bits>0 and pilot set is huge, subsamples deterministically.
"""
function scan_best_alignment_fast(xhat0::Vector{ComplexF64},
                                  b1024::Vector{Int};
                                  pilot_frac::Float64=0.10,
                                  σ2::Float64=0.5,
                                  Dmax::Int=25,
                                  scan_max_bits::Int=2000)

    pilot_bits = choose_pilots_bits(1024; frac=pilot_frac)

    if scan_max_bits > 0 && length(pilot_bits) > scan_max_bits
        step = max(1, length(pilot_bits) ÷ scan_max_bits)
        pilot_bits = pilot_bits[1:step:end]
        length(pilot_bits) > scan_max_bits && (pilot_bits = pilot_bits[1:scan_max_bits])
    end

    bestScore = -Inf
    bestD = 0
    bestRot = 1 + 0im
    bestConj = false

    for D in 0:Dmax
        xD = shift_left(xhat0, D)
        for rot in ROT_SET, cj in (false, true)
            xR = apply_rot_conj(xD, rot, cj)
            sc = pilot_score_from_xhat(xR, b1024, pilot_bits, σ2)
            if sc > bestScore
                bestScore = sc
                bestD = D
                bestRot = rot
                bestConj = cj
            end
        end
    end

    return bestScore, bestD, bestRot, bestConj, pilot_bits
end

export ROT_SET,
       apply_candidate,
       choose_pilots_bits,
       bits_to_syms_pilots,
       scan_best_alignment_fast,
       pilot_score_from_xhat

end # module

