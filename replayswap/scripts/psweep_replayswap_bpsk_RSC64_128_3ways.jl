#!/usr/bin/env julia
# scripts/psweep_replayswap_bpsk_RSC64_128_4ways.jl
#
# Pilot sweep on replay-swap dataset: RSC(64->128) BPSK (per-frame blocks).
#
# Methods (4 ways):
#   (A) EQ+BCJR (single-pass):  LMMSE deconv -> LLR(b128) -> clamp pilots -> RSC BCJR -> u64_hat
#   (B) JSDC+BCJR:             JSDC BPSK (λ_par=0) with ORACLE pilots + EQ prior -> LLR(b128)
#                              -> clamp pilots -> RSC BCJR -> u64_hat
#   (C) TurboEQ (iterative):   iterative EQ <-> BCJR loop (uses lib/TurboEQ_BPSK_RSC64_128.jl)
#                              -> outputs u64 and b128(post) metrics
#   (D) EQ-hard b128 baseline: hard decision on EQ LLRs (no clamp)
#
# Uses SAME selected blocks for all p (selected once by corr_thr + seed_sel).
#
# Output CSV:
#   data/runs/psweep_replayswap_bpsk_RSC64_128_4ways.csv
#
# Run:
#   julia --project=. scripts/psweep_replayswap_bpsk_RSC64_128_4ways.jl
#   julia --project=. scripts/psweep_replayswap_bpsk_RSC64_128_4ways.jl --psweep "0.0:0.05:0.5" --corr 0.10 --nblk 400
#
# Notes:
# - LLR convention: log P(b=0)/P(b=1), hard bit = (L < 0) ? 1 : 0
# - BPSK mapping in your dataset builder: bit1 -> +1, bit0 -> -1
#   => channel LLR log(P0/P1) ≈ -2*x_soft/σ2
# - Pilots are "oracle known" positions chosen evenly in 1..128 by frac p,
#   clamped into LLRs and also fed to JSDC as pilot targets.
#
using Random, Printf, Statistics, LinearAlgebra
using JLD2, DataFrames, CSV

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
const LS = ensure_linksim_loaded!()
using .LS: jsdc_qpsk_manual

# TurboEQ-style iterative EQ<->BCJR decoder (BPSK RSC 64->128)
include(joinpath(ROOT, "lib", "TurboEQ_BPSK_RSC64_128.jl"))
using .TurboEQ_BPSK_RSC64_128: decode_turboeq_rsc_bpsk

# ----------------------------
# CLI helpers
# ----------------------------
function parse_psweep(s::String)
    t = replace(strip(s), " " => "")
    isempty(t) && return Float64[]
    if occursin(":", t)
        parts = split(t, ":")
        if length(parts) == 2
            a = parse(Float64, parts[1]); b = parse(Float64, parts[2])
            step = 0.1
            return collect(a:step:b)
        elseif length(parts) == 3
            a = parse(Float64, parts[1]); step = parse(Float64, parts[2]); b = parse(Float64, parts[3])
            return collect(a:step:b)
        else
            error("Bad --psweep. Use a:b or a:step:b or comma list.")
        end
    else
        return Float64.(parse.(Float64, split(t, ",")))
    end
end

# ----------------------------
# Core utilities
# ----------------------------

@inline function conv_prefix(h::Vector{ComplexF64}, x::Vector{ComplexF64}, T::Int)
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

@inline function convmtx_prefix(h::Vector{ComplexF64}, T::Int)
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
    return M \ (Hc' * y)
end

function eq_lmmse_with_sigma2(y::Vector{ComplexF64}, h::Vector{ComplexF64};
                              σ2_init::Float64=0.30, iters::Int=2)
    σ2 = max(σ2_init, 1e-6)
    x = eq_lmmse_deconv(y, h, σ2)
    for _ in 1:iters
        yhat = conv_prefix(h, x, length(y))
        σ2 = clamp(Float64(mean(abs2, y .- yhat)), 1e-6, 10.0)
        x = eq_lmmse_deconv(y, h, σ2)
    end
    return x, σ2
end

# bit1 -> +1, bit0 -> -1
@inline bpsk_sym_from_bit(b::Int) = (b == 1 ? 1.0 : -1.0)

# LLR = log P0/P1, mapping bit1->+1,bit0->-1: L ≈ -2*x/σ2
@inline function bpsk_llr_logP0P1(x_soft::Vector{Float64}, σ2::Float64; clip::Float64=25.0)
    c = -2.0 / max(σ2, 1e-12)
    L = c .* x_soft
    return clamp.(L, -clip, clip)
end

@inline hard_from_llr(L::AbstractVector{<:Real}) = Int.(L .< 0)
@inline ber(a::Vector{Int}, b::Vector{Int}) = mean(a .!= b)
@inline psr_pkt(a::Vector{Int}, b::Vector{Int}) = all(a .== b) ? 1.0 : 0.0

# evenly spaced pilot bit positions in 1..n
function choose_pilots_bits(n::Int; frac::Float64)
    frac <= 0 && return Int[]
    Np = max(1, round(Int, frac*n))
    posf = collect(range(1, stop=n, length=Np))
    pos = unique!(clamp.(round.(Int, posf), 1, n))
    sort!(pos)
    return pos
end

# clamp pilots into LLR(log P0/P1): bit0 => +L, bit1 => -L
function clamp_pilots!(L::Vector{Float64}, btrue::Vector{Int}, pos::Vector{Int}; clampL::Float64=25.0)
    isempty(pos) && return L
    @inbounds for p in pos
        L[p] = (btrue[p] == 0) ? clampL : -clampL
    end
    return L
end

# sign/orientation fix for x_soft using pilot targets (BPSK symbols ±1)
function orient_by_pilots(x_soft::Vector{Float64}, pilot_pos::Vector{Int}, pilot_bpsk::Vector{Float64})
    isempty(pilot_pos) && return x_soft, +1
    v = 0.0
    @inbounds for (k, j) in enumerate(pilot_pos)
        v += x_soft[j] * pilot_bpsk[k]
    end
    if v < 0
        return (-x_soft), -1
    else
        return x_soft, +1
    end
end

# ----------------------------
# RSC(64->128) BCJR (log domain)
# Polys: feedback=7(111), feedforward=5(101), K=3 (4 states)
# Inputs: Lsys, Lpar are log(P0/P1), La is a priori on u bits (same convention)
# Output: Lu_post = log(P0/P1) for u bits
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
            f = u ⊻ s1 ⊻ s2          # feedback=111
            p = f ⊻ s2               # feedforward=101
            ns1 = f
            ns2 = s1
            ns = (ns1 << 1) | ns2
            next_state[s+1, u+1] = ns
            parity_bit[s+1, u+1] = p
        end
    end
end
__init_trellis__()

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
        c = maximum(@view α[t+1, :]); isfinite(c) && (α[t+1, :] .-= c)
    end

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
        c = maximum(@view β[t, :]); isfinite(c) && (β[t, :] .-= c)
    end

    Lu_post = Vector{Float64}(undef, N)
    @inbounds for t in 1:N
        num0 = -Inf
        num1 = -Inf
        for s in 0:3
            a = α[t, s+1]
            a == -Inf && continue
            for u in 0:1
                ns = next_state[s+1, u+1]
                p  = parity_bit[s+1, u+1]
                g = 0.5 * ( (1 - 2u) * (La[t] + Lsys[t]) + (1 - 2p) * Lpar[t] )
                val = a + g + β[t+1, ns+1]
                if u == 0
                    num0 = lse2(num0, val)
                else
                    num1 = lse2(num1, val)
                end
            end
        end
        Lu_post[t] = num0 - num1
    end
    return Lu_post
end

# ----------------------------
# JSDC(eq) LLR for b128 (λ_par=0)
# Uses oracle pilots (positions + targets) + EQ prior.
# Returns:
#   Lcode_logP0P1 (len 128), plus a hard bit vector from sign(m_final)
# ----------------------------
function jsdc_eq_bpsk_Lcode(y::Vector{ComplexF64}, h::Vector{ComplexF64}, σ2_init::Float64,
                            pilot_pos::Vector{Int}, pilot_bpsk::Vector{Float64};
                            λ_pil::Float64=20.0,
                            λ_prior::Float64=1.0,
                            η_z::Float64=3e-4,
                            max_iter::Int=600,
                            γ_z::Float64=5e-3,
                            γ_h::Float64=1e-3,
                            llr_clip::Float64=25.0)

    x_soft_c, σ2_hat = eq_lmmse_with_sigma2(y, h; σ2_init=σ2_init, iters=1)
    x0_raw = real.(x_soft_c)

    x0, _sgn = orient_by_pilots(x0_raw, pilot_pos, pilot_bpsk)

    # Prior in log(P1/P0) for JSDC internals: L10 ≈ +2*x/σ2
    L10 = clamp.((2.0 / max(σ2_hat, 1e-12)) .* x0, -llr_clip, llr_clip)

    m_init = clamp.(x0, -0.999, 0.999)
    z_init = atanh.(m_init)

    h_pos = collect(1:length(h))

    # Dummy "code" with only n needed; parity empty and λ_par=0
    dummy_code = (; n=length(y))
    parity_dummy = Vector{Vector{Int}}()

    xhat_bits, _hhat, info = jsdc_qpsk_manual(
        y, dummy_code, parity_dummy, pilot_pos, pilot_bpsk, h_pos;
        modulation=:bpsk,
        λ_par=0.0,
        λ_pil=λ_pil,
        γ_z=γ_z,
        γ_h=γ_h,
        η_z=η_z,
        η_h=0.0,
        max_iter=max_iter,
        h_init=h,
        z_init=z_init,
        L_prior=L10,
        λ_prior=λ_prior,
        σ2_data=σ2_hat,
        verbose=false
    )

    # IMPORTANT: jsdc_qpsk_manual should return info.z_final
    zf = Float64.(info.z_final)
    # Convert to Lcode log(P0/P1): L01 = -2*z  (since 2z ≈ log(P1/P0))
    Lcode = clamp.((-2.0) .* zf, -llr_clip, llr_clip)
    return Int.(xhat_bits), Lcode, σ2_hat
end

# ----------------------------
# Helpers for NaN-mean (because we store NaN when disabled)
# ----------------------------
@inline nanmean(v::AbstractVector{<:Real}) = mean(v[.!isnan.(v)])

# ----------------------------
# Main
# ----------------------------
function main()
    dataset = joinpath(DATA_DIR, "replayswap_bpsk_RSC_64_128_from_realdata_donorLS_h20_rho1e-2.jld2")
    outcsv  = joinpath(DATA_DIR, "runs", "psweep_replayswap_bpsk_RSC64_128_4ways.csv")

    # selection
    corr_thr = 0.10
    use_nblk = 900
    seed_sel = 12648430
    start = 1  # start within shuffled eligible list (1-based)

    # sweep
    psweep = collect(0.0:0.1:0.5)

    # EQ params
    σ2_init = 1.30
    eq_σ2_iters = 2
    llr_clip = 25.0
    clampL = 25.0

    # JSDC params
    jsdc_enable = true
    λ_pil = 60.0
    λ_prior = 1.0
    η_z = 3e-3
    max_iter = 600
    h_jsdc_Ls = 0  # 0 use full h; else window strongest-tap
    γ_z = 5e-2
    γ_h = 1e-2

    # TurboEQ params (iterative)
    turbo_enable = true
    turbo_iters = 2
    turbo_eq_σ2_iters = 1

    # CLI
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--dataset"; i+=1; dataset = ARGS[i]
        elseif a == "--outcsv"; i+=1; outcsv = ARGS[i]
        elseif a == "--corr"; i+=1; corr_thr = parse(Float64, ARGS[i])
        elseif a == "--nblk"; i+=1; use_nblk = parse(Int, ARGS[i])
        elseif a == "--seed_sel"; i+=1; seed_sel = parse(Int, ARGS[i])
        elseif a == "--start"; i+=1; start = parse(Int, ARGS[i])
        elseif a == "--psweep"; i+=1; psweep = parse_psweep(ARGS[i])

        elseif a == "--σ2_init"; i+=1; σ2_init = parse(Float64, ARGS[i])
        elseif a == "--eq_σ2_iters"; i+=1; eq_σ2_iters = parse(Int, ARGS[i])
        elseif a == "--llr_clip"; i+=1; llr_clip = parse(Float64, ARGS[i])
        elseif a == "--clampL"; i+=1; clampL = parse(Float64, ARGS[i])

        elseif a == "--jsdc_enable"; i+=1; jsdc_enable = (parse(Int, ARGS[i]) != 0)
        elseif a == "--lam_pil"; i+=1; λ_pil = parse(Float64, ARGS[i])
        elseif a == "--lam_prior"; i+=1; λ_prior = parse(Float64, ARGS[i])
        elseif a == "--etaz"; i+=1; η_z = parse(Float64, ARGS[i])
        elseif a == "--max_iter"; i+=1; max_iter = parse(Int, ARGS[i])
        elseif a == "--h_jsdc_Ls"; i+=1; h_jsdc_Ls = parse(Int, ARGS[i])
        elseif a == "--gamz"; i+=1; γ_z = parse(Float64, ARGS[i])
        elseif a == "--gamh"; i+=1; γ_h = parse(Float64, ARGS[i])

        elseif a == "--turbo_enable"; i+=1; turbo_enable = (parse(Int, ARGS[i]) != 0)
        elseif a == "--turbo_iters"; i+=1; turbo_iters = parse(Int, ARGS[i])
        elseif a == "--turbo_eq_σ2_iters"; i+=1; turbo_eq_σ2_iters = parse(Int, ARGS[i])
        else
            error("Unknown arg: $a")
        end
        i += 1
    end

    isfile(dataset) || error("Missing dataset: $dataset")
    mkpath(dirname(outcsv))

    d = JLD2.load(dataset)
    ymat = d["y_bpsk_swapped"]
    umat = d["u64_mat"]
    bmat = d["b128_mat"]
    hmat = d["h_blk_mat"]
    corr = d["corr_donor"]

    @assert size(ymat,2) == 128
    @assert size(umat,2) == 64
    @assert size(bmat,2) == 128

    eligible = findall(corr .>= corr_thr)
    isempty(eligible) && error("No eligible blocks at corr_thr=$corr_thr")
    rng = MersenneTwister(seed_sel)
    shuffle!(rng, eligible)

    start = clamp(start, 1, length(eligible))
    if use_nblk < 0
        blk_list = eligible[start:end]
    else
        blk_list = eligible[start : min(length(eligible), start + use_nblk - 1)]
    end
    isempty(blk_list) && error("Empty blk_list after selection")

    df = DataFrame(
        p=Float64[], blk=Int[], corr=Float64[], npil=Int[],
        sigma2_hat=Float64[],

        # EQ-hard b128 (no pilot clamp)
        eqhard_ber_b128=Float64[], eqhard_psr_b128=Float64[],

        # EQ+BCJR (single pass + clamp)
        eq_ber_u64=Float64[], eq_psr_u64=Float64[],
        eq_ber_b128=Float64[], eq_psr_b128=Float64[],

        # JSDC+BCJR (pilots inside JSDC + clamp)
        jsdc_ber_u64=Float64[], jsdc_psr_u64=Float64[],
        jsdc_ber_b128=Float64[], jsdc_psr_b128=Float64[],

        # TurboEQ (iterative EQ<->BCJR)
        turbo_ber_u64=Float64[], turbo_psr_u64=Float64[],
        turbo_ber_b128=Float64[], turbo_psr_b128=Float64[],
        turbo_ber_b128_ch=Float64[],
    )

    println("==============================================================")
    @printf("PSWEEP RSC(64->128) BPSK replay-swap | eligible=%d using=%d | corr_thr=%.2f start=%d\n",
            length(eligible), length(blk_list), corr_thr, start)
    @printf("psweep = %s\n", string(psweep))
    @printf("EQ: σ2_init=%.3f eq_σ2_iters=%d | JSDC=%s (λ_pil=%.1f λ_prior=%.1f ηz=%.1e it=%d h_jsdc_Ls=%d)\n",
            σ2_init, eq_σ2_iters, jsdc_enable ? "ON" : "OFF", λ_pil, λ_prior, η_z, max_iter, h_jsdc_Ls)
    @printf("TurboEQ: %s (turbo_iters=%d turbo_eq_σ2_iters=%d)\n",
            turbo_enable ? "ON" : "OFF", turbo_iters, turbo_eq_σ2_iters)
    println("==============================================================")

    for (pi, p) in enumerate(psweep)
        @printf("\n--- p=%.3f (%d/%d) ---\n", p, pi, length(psweep))

        pilot_pos = choose_pilots_bits(128; frac=p)

        for (ii, b) in enumerate(blk_list)
            y = ComplexF64.(vec(ymat[b, :]))
            u_true = Int.(vec(umat[b, :]))
            b_true = Int.(vec(bmat[b, :]))
            hfull = ComplexF64.(vec(hmat[b, :]))
            Lh = min(length(hfull), length(y))
            h_use = hfull[1:Lh]

            # oracle pilot targets from true coded bits
            pilot_bpsk = Float64[bpsk_sym_from_bit(b_true[j]) for j in pilot_pos]
            npil = length(pilot_pos)

            # ---------------- EQ (single pass) ----------------
            x_soft_c, σ2_hat = eq_lmmse_with_sigma2(y, h_use; σ2_init=σ2_init, iters=eq_σ2_iters)
            x_soft_raw = real.(x_soft_c)

            # orient EQ soft by oracle pilots (important when p>0)
            x_soft, _sgn = orient_by_pilots(x_soft_raw, pilot_pos, pilot_bpsk)

            Lcode_eq = bpsk_llr_logP0P1(x_soft, σ2_hat; clip=llr_clip)

            # EQ-hard baseline (no clamp)
            b_hat_eqhard = hard_from_llr(Lcode_eq)
            eqhard_ber_b128 = ber(b_hat_eqhard, b_true)
            eqhard_psr_b128 = psr_pkt(b_hat_eqhard, b_true)

            # EQ + BCJR with pilot clamp
            Lcode_eqc = copy(Lcode_eq)
            clamp_pilots!(Lcode_eqc, b_true, pilot_pos; clampL=clampL)

            b_hat_eq = hard_from_llr(Lcode_eqc)
            eq_ber_b128 = ber(b_hat_eq, b_true)
            eq_psr_b128 = psr_pkt(b_hat_eq, b_true)

            Lsys = Lcode_eqc[1:2:end]
            Lpar = Lcode_eqc[2:2:end]
            Lu_post = bcjr_rsc(Lsys, Lpar, zeros(Float64, 64))
            u_hat_eq = hard_from_llr(Lu_post)

            eq_ber_u64 = ber(u_hat_eq, u_true)
            eq_psr_u64 = psr_pkt(u_hat_eq, u_true)

            # ---------------- JSDC + BCJR ----------------
            jsdc_ber_u64 = NaN
            jsdc_psr_u64 = NaN
            jsdc_ber_b128 = NaN
            jsdc_psr_b128 = NaN

            if jsdc_enable
                # optional shorten h for JSDC
                h_js = h_use
                if h_jsdc_Ls > 0
                    mags = abs.(h_use)
                    ℓ0 = argmax(mags)
                    a = max(1, ℓ0 - (h_jsdc_Ls ÷ 2))
                    bb = min(Lh, a + h_jsdc_Ls - 1)
                    a = max(1, bb - h_jsdc_Ls + 1)
                    h_js = h_use[a:bb]
                end

                _bhat_bits, Lcode_jsdc, _σ2j = jsdc_eq_bpsk_Lcode(
                    y, h_js, σ2_hat, pilot_pos, pilot_bpsk;
                    λ_pil=λ_pil, λ_prior=λ_prior,
                    η_z=η_z, max_iter=max_iter,
                    γ_z=γ_z, γ_h=γ_h,
                    llr_clip=llr_clip
                )

                Lcode_jc = copy(Lcode_jsdc)
                clamp_pilots!(Lcode_jc, b_true, pilot_pos; clampL=clampL)

                b_hat_js = hard_from_llr(Lcode_jc)
                jsdc_ber_b128 = ber(b_hat_js, b_true)
                jsdc_psr_b128 = psr_pkt(b_hat_js, b_true)

                LsysJ = Lcode_jc[1:2:end]
                LparJ = Lcode_jc[2:2:end]
                Lu_postJ = bcjr_rsc(LsysJ, LparJ, zeros(Float64, 64))
                u_hat_js = hard_from_llr(Lu_postJ)

                jsdc_ber_u64 = ber(u_hat_js, u_true)
                jsdc_psr_u64 = psr_pkt(u_hat_js, u_true)
            end

            # ---------------- TurboEQ (iterative EQ <-> BCJR) ----------------
            turbo_ber_u64 = NaN
            turbo_psr_u64 = NaN
            turbo_ber_b128 = NaN
            turbo_psr_b128 = NaN
            turbo_ber_b128_ch = NaN

            if turbo_enable
                tout = decode_turboeq_rsc_bpsk(
                    y, h_use, u_true, b_true;
                    p=p,
                    turbo_iters=turbo_iters,
                    σ2_init=σ2_init,
                    eq_σ2_iters=turbo_eq_σ2_iters,
                    llr_clip=llr_clip
                )

                u_hat_t = tout.u64_hat
                b_hat_t_post = hard_from_llr(tout.llr128_post)
                b_hat_t_ch   = hard_from_llr(tout.llr128_ch)

                turbo_ber_u64 = ber(u_hat_t, u_true)
                turbo_psr_u64 = psr_pkt(u_hat_t, u_true)

                turbo_ber_b128 = ber(b_hat_t_post, b_true)
                turbo_psr_b128 = psr_pkt(b_hat_t_post, b_true)

                turbo_ber_b128_ch = ber(b_hat_t_ch, b_true)
            end

            push!(df, (
                p=p, blk=b, corr=Float64(corr[b]), npil=npil,
                sigma2_hat=σ2_hat,

                eqhard_ber_b128=eqhard_ber_b128,
                eqhard_psr_b128=eqhard_psr_b128,

                eq_ber_u64=eq_ber_u64,
                eq_psr_u64=eq_psr_u64,
                eq_ber_b128=eq_ber_b128,
                eq_psr_b128=eq_psr_b128,

                jsdc_ber_u64=jsdc_ber_u64,
                jsdc_psr_u64=jsdc_psr_u64,
                jsdc_ber_b128=jsdc_ber_b128,
                jsdc_psr_b128=jsdc_psr_b128,

                turbo_ber_u64=turbo_ber_u64,
                turbo_psr_u64=turbo_psr_u64,
                turbo_ber_b128=turbo_ber_b128,
                turbo_psr_b128=turbo_psr_b128,
                turbo_ber_b128_ch=turbo_ber_b128_ch,
            ))

            if ii == 1 || ii % 50 == 0 || ii == length(blk_list)
                @printf("  blk %d/%d | u64 PSR: EQ=%.2f JSDC=%.2f TURBO=%.2f | npil=%d\n",
                        ii, length(blk_list),
                        eq_psr_u64,
                        jsdc_enable ? jsdc_psr_u64 : NaN,
                        turbo_enable ? turbo_psr_u64 : NaN,
                        npil)
            end
        end

        sub = df[df.p .== p, :]

        eq_psr = mean(sub.eq_psr_u64)
        js_psr = jsdc_enable ? nanmean(sub.jsdc_psr_u64) : NaN
        tb_psr = turbo_enable ? nanmean(sub.turbo_psr_u64) : NaN

        eq_berm = mean(sub.eq_ber_u64)
        js_berm = jsdc_enable ? nanmean(sub.jsdc_ber_u64) : NaN
        tb_berm = turbo_enable ? nanmean(sub.turbo_ber_u64) : NaN

        @printf("  mean @ p=%.3f | u64 PSR: EQ=%.3f JSDC=%.3f TURBO=%.3f | u64 BER: EQ=%.4f JSDC=%.4f TURBO=%.4f | b128 PSR(EQhard)=%.3f\n",
                p, eq_psr, js_psr, tb_psr, eq_berm, js_berm, tb_berm, mean(sub.eqhard_psr_b128))
    end

    CSV.write(outcsv, df)
    println("\nSaved CSV → $outcsv")

    println("\n---------------- Overall means ----------------")
    @printf("EQ-hard b128: PSR(pkt)=%.4f BER=%.4f\n",
            mean(df.eqhard_psr_b128), mean(df.eqhard_ber_b128))
    @printf("EQ+BCJR  u64:  PSR(pkt)=%.4f BER=%.4f\n",
            mean(df.eq_psr_u64), mean(df.eq_ber_u64))
    if jsdc_enable && any(.!isnan.(df.jsdc_psr_u64))
        @printf("JSDC+BCJR u64: PSR(pkt)=%.4f BER=%.4f\n",
                nanmean(df.jsdc_psr_u64), nanmean(df.jsdc_ber_u64))
    end
    if turbo_enable && any(.!isnan.(df.turbo_psr_u64))
        @printf("TurboEQ  u64:  PSR(pkt)=%.4f BER=%.4f\n",
                nanmean(df.turbo_psr_u64), nanmean(df.turbo_ber_u64))
    end
end

main()
