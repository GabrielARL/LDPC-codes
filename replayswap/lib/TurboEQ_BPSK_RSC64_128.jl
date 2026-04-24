# lib/TurboEQ_BPSK_RSC64_128.jl
#
# TurboEQ-style decoder for replay-swap BPSK RSC(64->128) dataset.
# Loop:
#   LMMSE deconv -> LLR128 (BPSK) -> (optional pilot clamp) -> RSC BCJR
#   -> posterior-mean x -> refine σ² -> repeat
#
# Conventions:
#   - bit LLR = log P(b=0)/P(b=1)
#   - hard bit: b_hat = (L < 0) ? 1 : 0
#   - BPSK mapping: bit1 -> +1, bit0 -> -1
#
# OPTION A (no genie):
#   - Pass pilot positions + their known bit values explicitly:
#       pilots_pos::Vector{Int}
#       pilots_bits::Vector{Int} (same length; 0/1 bits)
#   - This file keeps a backward-compatible wrapper that still accepts b128_true,
#     but the "realistic" API does not require truth, only known pilots.
#
module TurboEQ_BPSK_RSC64_128

using LinearAlgebra, Statistics

export decode_turboeq_rsc_bpsk,
       choose_pilots_bits,
       hard_from_llr

# ----------------------------
# Small helpers
# ----------------------------
@inline hard_from_llr(L::AbstractVector{<:Real}) = Int.(L .< 0)

function choose_pilots_bits(n::Int; frac::Float64)
    frac <= 0 && return Int[]
    Np = max(1, round(Int, frac*n))
    posf = collect(range(1, stop=n, length=Np))
    pos = unique!(clamp.(round.(Int, posf), 1, n))
    sort!(pos)
    return pos
end

@inline function clamp_pilots_L01!(L01::Vector{Float64},
                                  pilot_bits::Vector{Int},
                                  pos::Vector{Int};
                                  clampL::Float64=25.0)
    @assert length(pilot_bits) == length(pos)
    @inbounds for (k, p) in enumerate(pos)
        b = pilot_bits[k]
        L01[p] = (b == 0) ? clampL : -clampL
    end
    return L01
end

# Prefix convolution: y[t] = Σ_{ℓ<=t} h[ℓ] x[t-ℓ+1]
function conv_prefix(h::Vector{ComplexF64}, x::Vector{ComplexF64}, T::Int)
    Lh = length(h)
    y = zeros(ComplexF64, T)
    @inbounds for t in 1:T
        acc = 0.0 + 0im
        for ℓ in 1:min(Lh, t)
            acc += h[ℓ] * x[t-ℓ+1]
        end
        y[t] = acc
    end
    return y
end

function convmtx_prefix(h::Vector{ComplexF64}, T::Int)
    Lh = length(h)
    H = zeros(ComplexF64, T, T)
    @inbounds for t in 1:T
        for k in 1:T
            ℓ = t - k + 1
            if 1 <= ℓ <= Lh
                H[t, k] = h[ℓ]
            end
        end
    end
    return H
end

function eq_lmmse_deconv(y::Vector{ComplexF64}, h::Vector{ComplexF64}, σ2::Float64)
    T = length(y)
    Hc = convmtx_prefix(h, T)
    M = Hc' * Hc
    @inbounds for i in 1:T
        M[i,i] += max(σ2, 1e-12)
    end
    rhs = Hc' * y
    return M \ rhs
end

function eq_lmmse_with_sigma2(y::Vector{ComplexF64},
                              h::Vector{ComplexF64};
                              σ2_init::Float64=0.30,
                              iters::Int=1)
    σ2 = max(σ2_init, 1e-6)
    x = eq_lmmse_deconv(y, h, σ2)
    for _ in 1:iters
        yhat = conv_prefix(h, x, length(y))
        σ2 = clamp(Float64(mean(abs2, y .- yhat)), 1e-6, 10.0)
        x = eq_lmmse_deconv(y, h, σ2)
    end
    return x, σ2
end

# BPSK LLR (log P0/P1) from soft symbols
@inline function bpsk_llr_logP0P1(x_soft::Vector{Float64}, σ2::Float64; clip::Float64=25.0)
    c = -2.0 / max(σ2, 1e-12)          # mapping bit1->+1 => LLR ≈ c*x
    L = c .* x_soft
    return clamp.(L, -clip, clip)
end

# Posterior mean for BPSK bits given L01 = log P0/P1
@inline bpsk_mean_from_L01(L01::Vector{Float64}) = -tanh.(0.5 .* L01)

# ----------------------------
# RSC BCJR (rate 1/2, K=3, 4-state)
# Trellis matches: FB=7 (111), P=5 (101)
# ----------------------------
@inline function lse2(a::Float64, b::Float64)
    if a == -Inf; return b; end
    if b == -Inf; return a; end
    m = max(a,b)
    return m + log(exp(a-m) + exp(b-m))
end

const NST = 4
const next_state = Array{Int}(undef, NST, 2)
const parity_bit = Array{Int}(undef, NST, 2)

function __init_trellis__()
    for s in 0:3
        s1 = (s >> 1) & 1
        s2 = s & 1
        for u in 0:1
            f = u ⊻ s1 ⊻ s2
            p = f ⊻ s2
            ns1 = f
            ns2 = s1
            ns = (ns1 << 1) | ns2
            next_state[s+1, u+1] = ns
            parity_bit[s+1, u+1] = p
        end
    end
    return nothing
end
__init_trellis__()

"""
bcjr_rsc(Lsys, Lpar, La)

Inputs are length-N LLRs (log P0/P1):
- Lsys[t] on systematic bit u[t]
- Lpar[t] on parity bit p[t]
- La[t]  a-priori on u[t]

Returns:
- Lu_post (LLR on u)
- Lpar_post (LLR on parity)
"""
function bcjr_rsc(Lsys::Vector{Float64}, Lpar::Vector{Float64}, La::Vector{Float64})
    N = length(Lsys)
    @assert length(Lpar) == N
    @assert length(La) == N

    α = fill(-Inf, N+1, NST)
    β = fill(-Inf, N+1, NST)
    α[1, 1] = 0.0
    @inbounds for s in 1:NST
        β[N+1, s] = 0.0
    end

    # forward
    @inbounds for t in 1:N
        for s in 0:3
            a = α[t, s+1]
            a == -Inf && continue
            for u in 0:1
                ns = next_state[s+1, u+1]
                p  = parity_bit[s+1, u+1]
                g = 0.5 * ( (1 - 2u) * (La[t] + Lsys[t]) + (1 - 2p) * Lpar[t] )
                α[t+1, ns+1] = lse2(α[t+1, ns+1], a + g)
            end
        end
        c = maximum(@view α[t+1, :])
        isfinite(c) && (α[t+1, :] .-= c)
    end

    # backward
    @inbounds for t in N:-1:1
        for s in 0:3
            acc = -Inf
            for u in 0:1
                ns = next_state[s+1, u+1]
                p  = parity_bit[s+1, u+1]
                g = 0.5 * ( (1 - 2u) * (La[t] + Lsys[t]) + (1 - 2p) * Lpar[t] )
                acc = lse2(acc, g + β[t+1, ns+1])
            end
            β[t, s+1] = acc
        end
        c = maximum(@view β[t, :])
        isfinite(c) && (β[t, :] .-= c)
    end

    Lu_post   = Vector{Float64}(undef, N)
    Lpar_post = Vector{Float64}(undef, N)

    @inbounds for t in 1:N
        num_u0 = -Inf; num_u1 = -Inf
        num_p0 = -Inf; num_p1 = -Inf
        for s in 0:3
            a = α[t, s+1]
            a == -Inf && continue
            for u in 0:1
                ns = next_state[s+1, u+1]
                p  = parity_bit[s+1, u+1]
                g = 0.5 * ( (1 - 2u) * (La[t] + Lsys[t]) + (1 - 2p) * Lpar[t] )
                val = a + g + β[t+1, ns+1]
                if u == 0
                    num_u0 = lse2(num_u0, val)
                else
                    num_u1 = lse2(num_u1, val)
                end
                if p == 0
                    num_p0 = lse2(num_p0, val)
                else
                    num_p1 = lse2(num_p1, val)
                end
            end
        end
        Lu_post[t]   = num_u0 - num_u1
        Lpar_post[t] = num_p0 - num_p1
    end

    return Lu_post, Lpar_post
end

# ----------------------------
# Public decode (Option A)
# ----------------------------
"""
decode_turboeq_rsc_bpsk(y, h;
                        p=0.20,
                        pilots_pos=Int[], pilots_bits=Int[],
                        turbo_iters=3, σ2_init=0.30, eq_σ2_iters=1, llr_clip=25.0,
                        u64_true=nothing, b128_true=nothing)

Option A (no genie): provide pilots explicitly:
  - pilots_pos: positions in 1:128 that are known
  - pilots_bits: the known 0/1 bits at those positions

If you DO NOT provide pilots_pos/pilots_bits, no pilot clamping is done.
If you provide p>0 but do not provide pilots_pos/pilots_bits, p is ignored.

Truth inputs are optional and used ONLY for metrics in the return tuple.
"""
function decode_turboeq_rsc_bpsk(y::Vector{ComplexF64},
                                 hfull::Vector{ComplexF64};
                                 p::Float64=0.20,
                                 pilots_pos::Vector{Int}=Int[],
                                 pilots_bits::Vector{Int}=Int[],
                                 turbo_iters::Int=3,
                                 σ2_init::Float64=0.30,
                                 eq_σ2_iters::Int=1,
                                 llr_clip::Float64=25.0,
                                 u64_true::Union{Nothing,Vector{Int}}=nothing,
                                 b128_true::Union{Nothing,Vector{Int}}=nothing)

    @assert length(y) == 128

    Lh = min(length(hfull), length(y))
    h = ComplexF64.(hfull[1:Lh])

    # Pilot handling (strict)
    if !isempty(pilots_pos) || !isempty(pilots_bits)
        isempty(pilots_pos)  && error("pilots_pos provided empty but pilots_bits not empty")
        isempty(pilots_bits) && error("pilots_bits missing: Option A requires known pilot bits")
        length(pilots_bits) == length(pilots_pos) || error("length(pilots_bits) != length(pilots_pos)")
        # keep within bounds + sorted
        @inbounds for k in eachindex(pilots_pos)
            (1 <= pilots_pos[k] <= 128) || error("pilot pos out of range: $(pilots_pos[k])")
            (pilots_bits[k] == 0 || pilots_bits[k] == 1) || error("pilot bit must be 0/1, got $(pilots_bits[k])")
        end
        sortperm_pos = sortperm(pilots_pos)
        pilots_pos  = pilots_pos[sortperm_pos]
        pilots_bits = pilots_bits[sortperm_pos]
    end

    σ2 = clamp(σ2_init, 1e-6, 10.0)
    llr128_ch = zeros(Float64, 128)
    llr128_post = zeros(Float64, 128)

    u64_hat = zeros(Int, 64)

    for _ in 1:max(1, turbo_iters)
        x_soft_c, σ2_hat = eq_lmmse_with_sigma2(y, h; σ2_init=σ2, iters=eq_σ2_iters)
        x_soft = real.(x_soft_c)

        llr128_ch = bpsk_llr_logP0P1(x_soft, σ2_hat; clip=llr_clip)

        # Option A: clamp from known pilots (no b128_true needed)
        if !isempty(pilots_pos)
            clamp_pilots_L01!(llr128_ch, pilots_bits, pilots_pos; clampL=llr_clip)
        end

        # BCJR on sys/par streams
        Lsys = llr128_ch[1:2:end]
        Lpar = llr128_ch[2:2:end]
        Lu_post, Lpar_post = bcjr_rsc(Lsys, Lpar, zeros(Float64, 64))
        u64_hat = hard_from_llr(Lu_post)

        # Build codeword posterior LLRs (sys/par interleaved) for posterior mean
        @inbounds for t in 1:64
            llr128_post[2t-1] = Lu_post[t]
            llr128_post[2t]   = Lpar_post[t]
        end

        # σ² refine using posterior mean codeword (no truth)
        xmean = ComplexF64.(bpsk_mean_from_L01(llr128_post))
        yhat = conv_prefix(h, xmean, length(y))
        σ2 = clamp(Float64(mean(abs2, y .- yhat)), 1e-6, 10.0)
    end

    # Metrics (truth optional; never used inside the loop)
    ber_u64 = isnothing(u64_true) ? NaN : mean(u64_hat .!= u64_true)
    preBER_b128 = isnothing(b128_true) ? NaN : mean(hard_from_llr(llr128_ch) .!= b128_true)

    return (
        u64_hat      = u64_hat,
        ber_u64      = ber_u64,
        llr128_ch    = llr128_ch,
        llr128_post  = llr128_post,
        sigma2_hat   = σ2,
        preBER_b128  = preBER_b128
    )
end

# ----------------------------
# Backward-compatible wrapper (your old call sites)
# ----------------------------
"""
decode_turboeq_rsc_bpsk(y, h, u64_true, b128_true; ...)

Compatibility wrapper.
- Uses b128_true ONLY to populate pilot bits for clamping at positions chosen by p.
- If in your experiment those bits are truly known pilots at the receiver, this is fine.
- If they are not, then calling this wrapper is indeed "pilot genie".
"""
function decode_turboeq_rsc_bpsk(y::Vector{ComplexF64},
                                 hfull::Vector{ComplexF64},
                                 u64_true::Vector{Int},
                                 b128_true::Vector{Int};
                                 p::Float64=0.20,
                                 turbo_iters::Int=3,
                                 σ2_init::Float64=0.30,
                                 eq_σ2_iters::Int=1,
                                 llr_clip::Float64=25.0)
    @assert length(u64_true) == 64
    @assert length(b128_true) == 128
    @assert length(y) == 128

    pilots_pos = choose_pilots_bits(128; frac=p)
    pilots_bits = isempty(pilots_pos) ? Int[] : Int.(b128_true[pilots_pos])

    return decode_turboeq_rsc_bpsk(
        y, hfull;
        p=p,
        pilots_pos=pilots_pos,
        pilots_bits=pilots_bits,
        turbo_iters=turbo_iters,
        σ2_init=σ2_init,
        eq_σ2_iters=eq_σ2_iters,
        llr_clip=llr_clip,
        u64_true=u64_true,
        b128_true=b128_true
    )
end

end # module
