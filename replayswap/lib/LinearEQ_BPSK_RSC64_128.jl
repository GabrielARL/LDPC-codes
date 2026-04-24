# lib/LinearEQ_BPSK_RSC64_128.jl
#
# “TurboEQ-lite” baseline for replay-swap BPSK RSC(64->128),
# but WITHOUT BCJR decoding. It uses a linear FIR LMMSE equalizer (Wiener filter)
# and then hard-decides the *systematic* bits only:
#   b128 = [u1,p1,u2,p2,...]  =>  u_hat from positions 1,3,5,...
#
# Optional: clamp known coded-bit pilots (Option A: pass pilots_pos + pilots_bits).
# Optional: iterate σ² by reconstructing yhat from soft mean symbols (no BCJR).
#
# Conventions:
#   - bit LLR = log P(b=0)/P(b=1)
#   - hard bit: b_hat = (L < 0) ? 1 : 0
#   - BPSK mapping: bit1 -> +1, bit0 -> -1
#
module LinearEQ_BPSK_RSC64_128

using LinearAlgebra, Statistics

export decode_lineareq_rsc_bpsk,
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

# BPSK LLR (log P0/P1) from soft symbols x_soft ≈ E[x] with x∈{-1,+1}
@inline function bpsk_llr_logP0P1(x_soft::Vector{Float64}, σ2_eff::Float64; clip::Float64=25.0)
    c = -2.0 / max(σ2_eff, 1e-12)   # bit1->+1 => LLR ≈ c*x
    L = c .* x_soft
    return clamp.(L, -clip, clip)
end

# Posterior mean symbol for BPSK given L01 = log P0/P1
@inline bpsk_mean_from_L01(L01::Vector{Float64}) = -tanh.(0.5 .* L01)

# ----------------------------
# FIR LMMSE (Wiener) equalizer
# ----------------------------
# Autocorr r[m] = E[y[t] conj(y[t-m])] for m>=0 with x white var=1
# r[m] = Σ_{k=m+1..Lh} h[k] conj(h[k-m])
function y_autocorr_from_h(h::Vector{ComplexF64}, M::Int)
    Lh = length(h)
    r = zeros(ComplexF64, M)  # r[1]=r[0], r[2]=r[1], ...
    @inbounds for m in 0:(M-1)
        acc = 0.0 + 0im
        for k in (m+1):Lh
            acc += h[k] * conj(h[k-m])
        end
        r[m+1] = acc
    end
    return r
end

# Build Toeplitz Ryy (M×M) from r[0:M-1] and add σ²I
function toeplitz_from_autocorr(r::Vector{ComplexF64}, σ2::Float64)
    M = length(r)
    R = Matrix{ComplexF64}(undef, M, M)
    @inbounds for i in 1:M
        for j in 1:M
            lag = i - j
            if lag >= 0
                R[i,j] = r[lag+1]
            else
                R[i,j] = conj(r[-lag+1])
            end
        end
    end
    @inbounds for i in 1:M
        R[i,i] += max(σ2, 1e-12)
    end
    return R
end

# Cross-corr p[i] = E[y[t-(i-1)] conj(x[t-D])] = h[D-(i-1)+1] if in range
function crosscorr_yx_from_h(h::Vector{ComplexF64}, M::Int, D::Int)
    Lh = length(h)
    p = zeros(ComplexF64, M)
    @inbounds for i in 1:M
        idx = D - (i-1) + 1  # 1-index into h
        p[i] = (1 <= idx <= Lh) ? h[idx] : (0.0 + 0im)
    end
    return p
end

# Compute Wiener taps w for estimating x[t-D] from [y[t],y[t-1],...,y[t-M+1]]
function fir_lmmse_taps(h::Vector{ComplexF64}, σ2::Float64; M::Int=15, D::Int=7)
    r = y_autocorr_from_h(h, M)
    R = toeplitz_from_autocorr(r, σ2)
    p = crosscorr_yx_from_h(h, M, D)
    w = R \ p
    return w
end

# Apply taps to get x_hat[t] for t=1..T using window ending at (t+D):
# window = [y[t+D], y[t+D-1], ..., y[t+D-M+1]]
function fir_apply(y::Vector{ComplexF64}, w::Vector{ComplexF64}; D::Int=7)
    T = length(y)
    M = length(w)
    xhat = zeros(ComplexF64, T)
    @inbounds for t in 1:T
        acc = 0.0 + 0im
        t0 = t + D
        for k in 1:M
            j = t0 - (k-1)
            yj = (1 <= j <= T) ? y[j] : (0.0 + 0im)
            acc += conj(w[k]) * yj
        end
        xhat[t] = acc
    end
    return xhat
end

# Calibrate equalizer output -> effective gain/noise on confident samples
function choose_pseudopilots(x_eq::Vector{Float64}; frac::Float64=0.25, minN::Int=16)
    T = length(x_eq)
    N = clamp(round(Int, frac*T), minN, T)
    idx = partialsortperm(abs.(x_eq), 1:N; rev=true)
    sort!(idx)
    return idx
end

function est_gain_sigma2_eff(x_eq::AbstractVector{<:Real},
                             idx::AbstractVector{<:Integer})
    isempty(idx) && return 1.0, clamp(Float64(var(Float64.(x_eq))), 1e-6, 10.0)

    n = length(idx)
    xp   = Vector{Float64}(undef, n)
    xe_f = Vector{Float64}(undef, n)

    @inbounds for k in 1:n
        v = Float64(x_eq[idx[k]])
        xe_f[k] = v
        xp[k] = (v >= 0) ? 1.0 : -1.0
    end

    g = dot(xp, xe_f) / max(dot(xp, xp), 1e-12)
    abs(g) < 1e-6 && (g = 1e-6)

    s = 0.0
    @inbounds for k in 1:n
        r = xe_f[k] - g * xp[k]
        s += r*r
    end

    σ2_eff = clamp(Float64(s / n), 1e-6, 10.0)
    return g, σ2_eff
end


# ----------------------------
# Public decode: linear equalizer only (no BCJR)
# ----------------------------
"""
decode_lineareq_rsc_bpsk(y, h;
                         pilots_pos=Int[], pilots_bits=Int[],
                         M=15, D=7,
                         iters=2, σ2_init=0.30,
                         eq_pp_frac=0.25,
                         llr_clip=25.0,
                         u64_true=nothing, b128_true=nothing)

- Builds FIR Wiener taps from h, σ²
- Equalizes y -> x_eq
- Calibrates to effective σ² on confident samples
- Converts to LLR128, clamps known pilots if provided
- Hard-decision systematic bits only => u64_hat = hard( LLR[1:2:end] )
- Optional σ² loop using soft mean symbols (no BCJR)

Truth inputs (u64_true/b128_true) are optional metrics only.
"""
function decode_lineareq_rsc_bpsk(y::Vector{ComplexF64},
                                  hfull::Vector{ComplexF64};
                                  pilots_pos::Vector{Int}=Int[],
                                  pilots_bits::Vector{Int}=Int[],
                                  M::Int=15,
                                  D::Int=7,
                                  iters::Int=2,
                                  σ2_init::Float64=0.30,
                                  eq_pp_frac::Float64=0.25,
                                  llr_clip::Float64=25.0,
                                  u64_true::Union{Nothing,Vector{Int}}=nothing,
                                  b128_true::Union{Nothing,Vector{Int}}=nothing)

    @assert length(y) == 128

    # Pilot validation
    if !isempty(pilots_pos) || !isempty(pilots_bits)
        isempty(pilots_pos)  && error("pilots_pos missing")
        isempty(pilots_bits) && error("pilots_bits missing")
        length(pilots_pos) == length(pilots_bits) || error("length mismatch pilots_pos/bits")
        @inbounds for k in eachindex(pilots_pos)
            (1 <= pilots_pos[k] <= 128) || error("pilot pos out of range: $(pilots_pos[k])")
            (pilots_bits[k] == 0 || pilots_bits[k] == 1) || error("pilot bit must be 0/1")
        end
        sp = sortperm(pilots_pos)
        pilots_pos  = pilots_pos[sp]
        pilots_bits = pilots_bits[sp]
    end

    Lh = min(length(hfull), 128)
    h = ComplexF64.(hfull[1:Lh])

    σ2 = clamp(σ2_init, 1e-6, 10.0)

    llr128_ch   = zeros(Float64, 128)
    llr128_post = zeros(Float64, 128)  # here “post” is just the clamped channel LLR (no decoder)
    u64_hat     = zeros(Int, 64)

    for _ in 1:max(1, iters)
        # Wiener FIR taps and equalization
        w = fir_lmmse_taps(h, σ2; M=M, D=D)
        xhat = fir_apply(y, w; D=D)
        x_eq = real.(xhat)

        # Calibrate gain + σ²_eff on confident samples (receiver-only)
        pp = choose_pseudopilots(x_eq; frac=eq_pp_frac, minN=max(16, M))
        g_eff, σ2_eff = est_gain_sigma2_eff(x_eq, pp)
        x_cal = x_eq ./ g_eff
        σ2_cal = σ2_eff / (g_eff^2)

        # LLR + pilot clamp (no truth needed if pilots_bits provided)
        llr128_ch = bpsk_llr_logP0P1(x_cal, σ2_cal; clip=llr_clip)
        if !isempty(pilots_pos)
            clamp_pilots_L01!(llr128_ch, pilots_bits, pilots_pos; clampL=llr_clip)
        end
        llr128_post .= llr128_ch

        # Hard-decision systematic bits only (u are odd positions)
        u64_hat = hard_from_llr(@view llr128_post[1:2:end])

        # Optional σ² refinement using soft mean symbols (still no BCJR)
        xmean = ComplexF64.(bpsk_mean_from_L01(llr128_post))
        yhat = conv_prefix(h, xmean, length(y))
        σ2 = clamp(Float64(mean(abs2, y .- yhat)), 1e-6, 10.0)
    end

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

end # module
