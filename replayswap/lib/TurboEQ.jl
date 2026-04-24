# DFEC/lib/TurboEQ.jl
#
# Lean TurboEQ (serial concat RSC) + LMMSE-FF front-end
# FIX INCLUDED: pad/truncate equalizer output so llr1024 is ALWAYS length 1024
#
# Conventions (matches your DFEC helpers):
#   - Bit LLR is log(P(b=0)/P(b=1))
#   - Hard decision: b_hat = (L < 0) ? 1 : 0
#   - QPSK axis mapping in your world: bit=1 -> +1, bit=0 -> -1

module TurboEQ

using LinearAlgebra, Statistics, Random, Printf

export decode_turboeq,
       choose_pilots_bits,
       qpsk_llr_from_xhat,
       hard_from_llr

# ------------------------------------------------------------------
# Small utilities
# ------------------------------------------------------------------

const CLAMP_L_DEFAULT = 25.0

@inline hardbit(L::Real) = (L < 0) ? 1 : 0
hard_from_llr(L::AbstractVector{<:Real}) = Int.(L .< 0)

# evenly spaced pilot bit positions in 1:n
function choose_pilots_bits(n::Int; frac::Float64)
    frac <= 0 && return Int[]
    Np = max(1, round(Int, frac*n))
    posf = collect(range(1, stop=n, length=Np))
    pos = unique!(clamp.(round.(Int, posf), 1, n))
    sort!(pos)
    return pos
end

@inline function clamp_pilots!(L::Vector{Float64}, btrue::Vector{Int}, pos::Vector{Int};
                              clampL::Float64=CLAMP_L_DEFAULT)
    @inbounds for p in pos
        L[p] = (btrue[p] == 0) ? clampL : -clampL
    end
    return L
end

# prefix convolution (no DSP dependency)
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

# QPSK soft LLR from equalized symbols (treat I/Q as independent BPSK)
function qpsk_llr_from_xhat(xhat::Vector{ComplexF64}, σ2::Float64)
    Ns = length(xhat)
    σ2eff = max(σ2, 1e-12)
    c = -2.0 / σ2eff
    L = Vector{Float64}(undef, 2*Ns)
    @inbounds for k in 1:Ns
        L[2k-1] = c * real(xhat[k])
        L[2k]   = c * imag(xhat[k])
    end
    return L
end

# log-sum-exp for a small vector
@inline function lse2(a::Float64, b::Float64)
    if a == -Inf; return b; end
    if b == -Inf; return a; end
    m = max(a,b)
    return m + log(exp(a-m) + exp(b-m))
end

# ------------------------------------------------------------------
# Minimal LMMSE-FF / equalize (self-contained fallback)
# If your Main.LMMSE exists, we will use it; otherwise we fall back.
# ------------------------------------------------------------------

# fallback: simple ZF-ish feedforward from first M taps (stable-ish)
function _fallback_wff(h::Vector{ComplexF64}, σ2::Float64; M::Int=11)
    M = min(M, length(h))
    ht = h[1:M]
    # matched filter normalized (not true LMMSE, but fine for plumbing)
    w = conj.(reverse(ht))
    s = sum(abs2, ht) + σ2
    return w ./ max(s, 1e-12)
end

# valid convolution (length(y)-M+1)
function _fallback_equalize_soft(y::Vector{ComplexF64}, w::Vector{ComplexF64})
    T = length(y)
    M = length(w)
    outlen = max(T - M + 1, 0)
    z = Vector{ComplexF64}(undef, outlen)
    @inbounds for t in 1:outlen
        acc = 0.0 + 0im
        for k in 1:M
            acc += w[k] * y[t + k - 1]
        end
        z[t] = acc
    end
    return z
end

# dispatch to your LMMSE module if present
function _lmmse_frontend(yr::Vector{ComplexF64}, h::Vector{ComplexF64}, σ2_init::Float64; M_eq::Int=11)
    if isdefined(Main, :LMMSE) &&
       isdefined(Main.LMMSE, :lmmse_ff) &&
       isdefined(Main.LMMSE, :equalize_soft)
        # expected API: wff, diag = lmmse_ff(h, σ2; M, D)
        wff, _ = Main.LMMSE.lmmse_ff(h, σ2_init; M=min(M_eq, max(3, length(yr))), D=0)
        z = Main.LMMSE.equalize_soft(yr, wff)
        return ComplexF64.(z)
    else
        wff = _fallback_wff(h, σ2_init; M=M_eq)
        z = _fallback_equalize_soft(yr, wff)
        return ComplexF64.(z)
    end
end

# ------------------------------------------------------------------
# RSC (rate-1/2 systematic) BCJR
# Polynomials: feedback=7 (111), feedforward=5 (101), K=3 (4 states)
# State = (s1,s2) bits packed in 0..3 where s1 is MSB.
# ------------------------------------------------------------------

# trellis tables
const NST = 4
const next_state = Array{Int}(undef, NST, 2)
const parity_bit = Array{Int}(undef, NST, 2)

function __init_trellis__()
    for s in 0:3
        s1 = (s >> 1) & 1
        s2 = s & 1
        for u in 0:1
            f = u ⊻ s1 ⊻ s2          # feedback=111
            p = f ⊻ s2               # feedforward=101 (f ⊻ s2)
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

# BCJR in log domain.
# Inputs are per-time-step LLRs:
#   Lsys[t] = log(P(sys=0)/P(sys=1))
#   Lpar[t] = log(P(par=0)/P(par=1))
#   La[t]   = apriori on input bit u[t] in the same LLR convention
#
# Returns:
#   Lu_post, Lu_ext, Lsys_post, Lpar_post, Lcode_post, Lcode_ext
function bcjr_rsc(Lsys::Vector{Float64}, Lpar::Vector{Float64}, La::Vector{Float64})
    N = length(Lsys)
    @assert length(Lpar) == N
    @assert length(La) == N

    α = fill(-Inf, N+1, NST)
    β = fill(-Inf, N+1, NST)
    α[1, 1] = 0.0                 # start state = 0
    @inbounds for s in 1:NST
        β[N+1, s] = 0.0           # no termination assumption (uniform)
    end

    # forward
    @inbounds for t in 1:N
        for s in 0:3
            a = α[t, s+1]
            a == -Inf && continue
            for u in 0:1
                ns = next_state[s+1, u+1]
                p  = parity_bit[s+1, u+1]
                # branch metric: (1-2b)*L/2 summed for u (prior+sys) and parity
                g = 0.5 * ( (1 - 2u) * (La[t] + Lsys[t]) + (1 - 2p) * Lpar[t] )
                α[t+1, ns+1] = lse2(α[t+1, ns+1], a + g)
            end
        end
    end

    # backward
    @inbounds for t in N:-1:1
        for s in 0:3
            acc = -Inf
            # β[t,s] = logsum_u exp( g + β[t+1,ns] )
            for u in 0:1
                ns = next_state[s+1, u+1]
                p  = parity_bit[s+1, u+1]
                g = 0.5 * ( (1 - 2u) * (La[t] + Lsys[t]) + (1 - 2p) * Lpar[t] )
                acc = lse2(acc, g + β[t+1, ns+1])
            end
            β[t, s+1] = acc
        end
    end

    Lu_post  = Vector{Float64}(undef, N)
    Lpar_post = Vector{Float64}(undef, N)

    # posteriors
    @inbounds for t in 1:N
        num_u0 = -Inf
        num_u1 = -Inf
        num_p0 = -Inf
        num_p1 = -Inf
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

    # extrinsic on u (systematic u is observed by Lsys)
    Lu_ext = Lu_post .- La .- Lsys
    # codeword post/ext (interleaved sys/par: [u1,p1,u2,p2,...])
    Lcode_post = Vector{Float64}(undef, 2N)
    Lcode_ext  = Vector{Float64}(undef, 2N)
    @inbounds for t in 1:N
        Lcode_post[2t-1] = Lu_post[t]
        Lcode_post[2t]   = Lpar_post[t]
        Lcode_ext[2t-1]  = Lu_ext[t]
        Lcode_ext[2t]    = Lpar_post[t] - Lpar[t]   # parity extrinsic
    end

    return Lu_post, Lu_ext, Lpar_post, Lcode_post, Lcode_ext
end

# ------------------------------------------------------------------
# Public: decode_turboeq
# ------------------------------------------------------------------
"""
decode_turboeq(yr, u256_true, b512_true, b512_i_true, b1024_true, hr_full, itlv;
               p=0.40, niters=3, damp=0.6, M_eq=11, σ2_init=0.5)

Returns a NamedTuple with:
  :u256_hat, :ber_u256
  :llr1024_ch, :llr1024_post
  :sigma2_hat, :preBER_b1024
"""
function decode_turboeq(yr::Vector{ComplexF64},
                        u256_true::Vector{Int},
                        b512_true::Vector{Int},
                        b512_i_true::Vector{Int},
                        b1024_true::Vector{Int},
                        hr_full::Vector{ComplexF64},
                        itlv::@NamedTuple{use_itlv::Bool, π::Vector{Int}, πinv::Vector{Int}};
                        p::Float64=0.40,
                        niters::Int=3,
                        damp::Float64=0.6,
                        M_eq::Int=11,
                        σ2_init::Float64=0.5)

    @assert iseven(length(b1024_true))
    nsym = length(b1024_true) ÷ 2   # should be 512
    T = length(yr)

    # ---- channel slice ----
    Lh = min(length(hr_full), T)
    h = hr_full[1:Lh]

    # ---- front-end equalize ----
    z = _lmmse_frontend(yr, h, σ2_init; M_eq=M_eq)

    # ---- FIX: pad/truncate to nsym so llr1024 is exactly 1024 ----
    if length(z) < nsym
        z = vcat(z, zeros(ComplexF64, nsym - length(z)))
    elseif length(z) > nsym
        z = z[1:nsym]
    end

    # ---- sigma2 estimate (prefix conv model) ----
    yhat = conv_prefix(h, z, T)
    σ2_hat = clamp(Float64(mean(abs2, yr .- yhat)), 1e-6, 10.0)

    llr1024_ch = qpsk_llr_from_xhat(z, σ2_hat)

    # pilots on inner code bits
    pilots_inner = choose_pilots_bits(length(llr1024_ch); frac=p)
    clamp_pilots!(llr1024_ch, b1024_true, pilots_inner)

    preBER_b1024 = mean(hard_from_llr(llr1024_ch) .!= b1024_true)

    # ---- Turbo loop: Inner (512->1024) <-> Outer (256->512) ----
    # codeword ordering assumption:
    #   inner b1024 = [sys1,par1, sys2,par2, ...]  (len 1024 from 512 inputs)
    #   outer b512  = [sys1,par1, sys2,par2, ...]  (len 512  from 256 inputs)

    # split inner sys/par LLRs (per time step)
    Lsys_in = llr1024_ch[1:2:end]
    Lpar_in = llr1024_ch[2:2:end]

    # prior on inner INPUT bits (which are b512_i)
    La_in = zeros(Float64, length(b512_i_true))   # 512

    # placeholders
    llr1024_post = copy(llr1024_ch)
    Lu_post_outer = zeros(Float64, length(u256_true))

    # pilots on outer codeword bits (work in NON-interleaved b512 space)
    pilots_outer_code = choose_pilots_bits(length(b512_true); frac=p)

    for it in 1:niters
        # ---- INNER BCJR ----
        Lu_post_in, Lu_ext_in, _, _, _ = bcjr_rsc(Lsys_in, Lpar_in, La_in)

        # pass extrinsic on inner input bits to OUTER (deinterleave if needed)
        L_to_outer_i = Lu_ext_in
        L_to_outer = itlv.use_itlv ? L_to_outer_i[itlv.πinv] : L_to_outer_i

        # clamp "outer pilots" using truth b512_true (still code-bit domain)
        L_to_outer = Vector{Float64}(L_to_outer)  # ensure mutable
        clamp_pilots!(L_to_outer, b512_true, pilots_outer_code)

        # outer sys/par LLRs
        Lsys_o = L_to_outer[1:2:end]
        Lpar_o = L_to_outer[2:2:end]
        La_u_outer = zeros(Float64, length(u256_true))  # no extra prior

        # ---- OUTER BCJR ----
        Lu_post_outer, _, _, Lcode_post_o, Lcode_ext_o = bcjr_rsc(Lsys_o, Lpar_o, La_u_outer)

        # feed back extrinsic on OUTER CODEWORD bits to INNER as prior (interleave if needed)
        La_next = itlv.use_itlv ? Lcode_ext_o[itlv.π] : Lcode_ext_o

        # damping
        @. La_in = (1 - damp) * La_in + damp * La_next
    end

    # build llr1024_post for reporting: re-run inner with final La_in and return codeword post
    Lu_post_in, _, _, Lcode_post_in, _ = bcjr_rsc(Lsys_in, Lpar_in, La_in)
    llr1024_post = copy(llr1024_ch)
    @inbounds for k in 1:length(llr1024_post)
        llr1024_post[k] = (k <= length(Lcode_post_in)) ? Lcode_post_in[k] : llr1024_post[k]
    end
    clamp_pilots!(llr1024_post, b1024_true, pilots_inner)

    u256_hat = hard_from_llr(Lu_post_outer)
    ber_u256 = mean(u256_hat .!= u256_true)

    return (
        u256_hat      = u256_hat,
        ber_u256      = ber_u256,
        llr1024_ch    = llr1024_ch,
        llr1024_post  = llr1024_post,
        sigma2_hat    = σ2_hat,
        preBER_b1024  = preBER_b1024
    )
end

end # module TurboEQ
