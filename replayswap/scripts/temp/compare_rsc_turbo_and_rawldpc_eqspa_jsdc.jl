#!/usr/bin/env julia
# scripts/compare_rsc_turbo_and_rawldpc_eqspa_jsdc.jl
#
# ONE script, TWO datasets:
#
# 1) RSC replay-swap BPSK dataset:
#    - decode with Turbo (EQ <-> BCJR) loop, output u64 metrics + b128 metrics
#
# 2) RAW donor LDPC BPSK dataset:
#    - decode with EQ+SPA and JSDC, output cw128 BER metrics
#
# Run:
#   julia --project=. scripts/compare_rsc_turbo_and_rawldpc_eqspa_jsdc.jl
#
# Outputs:
#   data/runs/compare_replayswap_bpsk_RSC64_128_TURBO.csv
#   data/runs/compare_raw_ldpc_EQSPA_JSDC.csv
#
# Notes:
# - LLR convention for BCJR: L01 = log P(b=0)/P(b=1) => bit=1 iff L01 < 0
# - BPSK mapping: bit1 -> +1, bit0 -> -1
# - For that mapping, channel L01 ≈ -2*x_soft/σ2

using Random, Printf, Statistics, LinearAlgebra
using JLD2, DataFrames, CSV
using SparseArrays

# RAW donor helpers need these (same as your compare_raw_donor_3ways.jl)
using DSP
using SignalAnalysis

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
const LS = ensure_linksim_loaded!()

using .LS: initcode, encode, get_H_sparse, sum_product_decode, jsdc_qpsk_manual, Code

# ============================================================
# Common small utils
# ============================================================

@inline hard_from_llr(L::AbstractVector{<:Real}) = Int.(L .< 0)
@inline ber(a::AbstractVector{Int}, b::AbstractVector{Int}) = mean(a .!= b)
@inline psr_pkt(a::AbstractVector{Int}, b::AbstractVector{Int}) = all(a .== b) ? 1.0 : 0.0

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

# LMMSE deconv (prefix model) with σ2 regularization
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

# iterative σ² estimation: solve x, then σ² = mean|y - conv(h,x)|²
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

# L01 = log(P0/P1) for mapping bit1->+1, bit0->-1 : L01 ≈ -2*x/σ2
@inline function bpsk_llr_logP0P1(x_soft::Vector{Float64}, σ2::Float64; clip::Float64=25.0)
    c = -2.0 / max(σ2, 1e-12)
    L = c .* x_soft
    return clamp.(L, -clip, clip)
end

# posterior mean for BPSK symbol s∈{-1,+1} from L01:
# E[s] = P(1)-P(0) = -tanh(L01/2)
@inline bpsk_mean_from_L01(L01::Vector{Float64}) = -tanh.(0.5 .* L01)

# ============================================================
# RSC BCJR (log domain), K=3, 4 states, polys fb=7 (111) ff=5 (101)
# Inputs: Lsys, Lpar, La in L01 convention
# Outputs: Lu_post (L01 on u), Lcode_post (L01 on code bits), Lcode_ext (extrinsic L01)
# ============================================================

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
end
__init_trellis__()

function bcjr_rsc_full(Lsys::Vector{Float64}, Lpar::Vector{Float64}, La::Vector{Float64})
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
        c = maximum(@view α[t+1, :]); isfinite(c) && (α[t+1, :] .-= c)
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
        c = maximum(@view β[t, :]); isfinite(c) && (β[t, :] .-= c)
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

    # extrinsic relative to channel inputs
    Lu_ext   = Lu_post .- La .- Lsys
    Lpar_ext = Lpar_post .- Lpar

    Lcode_post = Vector{Float64}(undef, 2N)
    Lcode_ext  = Vector{Float64}(undef, 2N)
    @inbounds for t in 1:N
        Lcode_post[2t-1] = Lu_post[t]
        Lcode_post[2t]   = Lpar_post[t]
        Lcode_ext[2t-1]  = Lu_ext[t]
        Lcode_ext[2t]    = Lpar_ext[t]
    end

    return Lu_post, Lcode_post, Lcode_ext
end

# ============================================================
# RSC dataset: Turbo decode (EQ <-> BCJR)
# ============================================================

function decode_rsc_turbo_eq_bcjr(y::Vector{ComplexF64}, h::Vector{ComplexF64};
                                  σ2_init::Float64=0.30,
                                  eq_σ2_iters::Int=1,
                                  turbo_iters::Int=4,
                                  llr_clip::Float64=25.0)
    σ2 = σ2_init
    u_hat = zeros(Int, 64)
    L01_last = zeros(Float64, 128)

    for _ in 1:max(1, turbo_iters)
        x_soft_c, σ2_hat = eq_lmmse_with_sigma2(y, h; σ2_init=σ2, iters=eq_σ2_iters)
        x_soft = real.(x_soft_c)

        L01 = bpsk_llr_logP0P1(x_soft, σ2_hat; clip=llr_clip)
        Lsys = L01[1:2:end]
        Lpar = L01[2:2:end]
        Lu_post, Lcode_post, _Lcode_ext = bcjr_rsc_full(Lsys, Lpar, zeros(Float64, 64))
        u_hat = hard_from_llr(Lu_post)

        # refine σ² from residual using posterior mean code bits
        xmean = bpsk_mean_from_L01(Lcode_post)
        yhat  = conv_prefix(h, ComplexF64.(xmean), length(y))
        σ2 = clamp(Float64(mean(abs2, y .- yhat)), 1e-6, 10.0)

        L01_last = L01
    end

    return u_hat, L01_last, σ2
end

# ============================================================
# RAW donor LDPC dataset: EQ+SPA and JSDC
# ============================================================

modulate_bpsk_bit(x::Int) = x == 1 ? (1.0 + 0im) : (-1.0 + 0im)

function extract_symbol_rate(row::AbstractVector{T}, T_frame::Int) where {T<:Number}
    @assert length(row) ≥ T_frame + 1
    return ComplexF64.(row[2:T_frame+1])
end

function shift_left(y::Vector{ComplexF64}, D::Int)
    T = length(y)
    D <= 0 && return y
    D >= T && return zeros(ComplexF64, T)
    return vcat(y[D+1:end], zeros(ComplexF64, D))
end

function ridge_ls_h(x::Vector{ComplexF64}, y::Vector{ComplexF64}, h_len::Int, rho::Float64)
    T = length(x)
    X = zeros(ComplexF64, T, h_len)
    @inbounds for t in 1:T
        for k in 1:h_len
            idx = t - k + 1
            X[t, k] = idx >= 1 ? x[idx] : (0.0 + 0im)
        end
    end
    A = X'X
    @inbounds for i in 1:h_len
        A[i,i] += rho
    end
    b = X'y
    return Vector{ComplexF64}(A \ b)
end

function rebuild_x_old_and_cw_mat(codeB::Code; T_frame::Int, num_data::Int, num_repeats::Int)
    x_mat  = zeros(ComplexF64, num_data, T_frame)
    cw_mat = zeros(Int,        num_data, T_frame)

    for i in 1:num_data
        bseq   = mseq(11)[i : (codeB.k + i - 1)]
        d_test = Int.((bseq .+ 1) ./ 2)  # {-1,+1} -> {0,1}
        cw     = encode(codeB, d_test)   # BitVector length n
        bits   = Int.(cw)
        @inbounds for t in 1:T_frame
            cw_mat[i, t] = bits[t]
            x_mat[i, t]  = modulate_bpsk_bit(bits[t])
        end
    end

    return repeat(x_mat,  num_repeats, 1),
           repeat(cw_mat, num_repeats, 1)
end

function build_parity_indices(H::SparseMatrixCSC{Bool, Int})
    m, _n = size(H)
    pi = [Int[] for _ in 1:m]
    I, J, _ = findnz(H)
    @inbounds for (i, j) in zip(I, J)
        push!(pi[i], j)
    end
    return pi
end

# SPA wrapper: negate x_soft because your SPA hard decision uses bit=1 when L_post<0
@inline function spa_from_soft(codeB::Code,
                               colsB::Vector{Vector{Int}},
                               parityB::Vector{Vector{Int}},
                               x_soft::Vector{Float64},
                               σ2::Float64;
                               max_iter::Int=50)
    H = get_H_sparse(codeB)
    y_for_spa = -x_soft
    x_hat, iters = sum_product_decode(H, y_for_spa, max(σ2, 1e-6), parityB, colsB; max_iter=max_iter)
    return Int.(x_hat), iters
end

# ============================================================
# Main
# ============================================================

function main()
    # --- RSC replay-swap dataset ---
    rsc_path = joinpath(DATA_DIR, "replayswap_bpsk_RSC_64_128_from_realdata_donorLS_h20_rho1e-2.jld2")
    out_rsc  = joinpath(DATA_DIR, "runs", "compare_replayswap_bpsk_RSC64_128_TURBO.csv")

    corr_thr = 0.10
    nblk_rsc = 200
    seed_sel = 12648430
    start    = 1

    # Turbo (EQ<->BCJR)
    σ2_init_rsc   = 0.30
    eq_σ2_iters   = 1
    turbo_iters   = 4
    llr_clip_rsc  = 25.0

    # --- RAW donor dataset + cache ---
    raw_path   = joinpath(DATA_DIR, "raw", "logged_packets_and_ytrain.jld2")
    cache_path = joinpath(DATA_DIR, "ls_cache_h20_rho1e-02_bestD.jld2")
    out_raw    = joinpath(DATA_DIR, "runs", "compare_raw_ldpc_EQSPA_JSDC.csv")

    nframes_use = 200  # -1 = all

    # must match your capture generation
    npc = 4
    T_frame = 128
    num_data = 20
    num_repeats = 45

    # per-frame channel estimation
    h_len = 20
    rho_ls = 1e-2

    # JSDC knobs
    jsdc_max_iter = 200
    jsdc_ηz = 1e-3
    jsdc_λpar = 0.1
    jsdc_γz = 5e-3
    jsdc_γh = 1e-3
    jsdc_λprior = 0.5

    # CLI
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--rsc"; i+=1; rsc_path = ARGS[i]
        elseif a == "--out_rsc"; i+=1; out_rsc = ARGS[i]
        elseif a == "--corr"; i+=1; corr_thr = parse(Float64, ARGS[i])
        elseif a == "--nblk_rsc"; i+=1; nblk_rsc = parse(Int, ARGS[i])
        elseif a == "--seed_sel"; i+=1; seed_sel = parse(Int, ARGS[i])
        elseif a == "--start"; i+=1; start = parse(Int, ARGS[i])

        elseif a == "--turbo_iters"; i+=1; turbo_iters = parse(Int, ARGS[i])
        elseif a == "--σ2_init_rsc"; i+=1; σ2_init_rsc = parse(Float64, ARGS[i])

        elseif a == "--raw"; i+=1; raw_path = ARGS[i]
        elseif a == "--cache"; i+=1; cache_path = ARGS[i]
        elseif a == "--out_raw"; i+=1; out_raw = ARGS[i]
        elseif a == "--nframes"; i+=1; nframes_use = parse(Int, ARGS[i])

        elseif a == "--jsdc_max_iter"; i+=1; jsdc_max_iter = parse(Int, ARGS[i])
        elseif a == "--jsdc_etaz"; i+=1; jsdc_ηz = parse(Float64, ARGS[i])
        elseif a == "--jsdc_lampar"; i+=1; jsdc_λpar = parse(Float64, ARGS[i])
        elseif a == "--jsdc_lamprior"; i+=1; jsdc_λprior = parse(Float64, ARGS[i])
        else
            error("Unknown arg: $a")
        end
        i += 1
    end

    mkpath(dirname(out_rsc))
    mkpath(dirname(out_raw))

    # ============================================================
    # Part 1: RSC replay-swap TURBO decode
    # ============================================================
    isfile(rsc_path) || error("Missing RSC dataset: $rsc_path")

    dr = JLD2.load(rsc_path)
    ymat = dr["y_bpsk_swapped"]
    umat = dr["u64_mat"]
    bmat = dr["b128_mat"]
    hmat = dr["h_blk_mat"]
    corr = dr["corr_donor"]

    nblk_all = size(ymat, 1)
    @assert size(ymat,2) == 128
    @assert size(umat,2) == 64
    @assert size(bmat,2) == 128
    @assert size(hmat,1) == nblk_all

    eligible = findall(corr .>= corr_thr)
    isempty(eligible) && error("No eligible RSC blocks at corr_thr=$corr_thr")

    rng = MersenneTwister(seed_sel)
    shuffle!(rng, eligible)

    start = clamp(start, 1, length(eligible))
    blk_list = eligible[start : min(length(eligible), start + nblk_rsc - 1)]
    isempty(blk_list) && error("Empty blk_list after selection")

    df_rsc = DataFrame(
        blk=Int[], corr=Float64[],
        turbo_ber_u64=Float64[], turbo_psr_u64=Float64[],
        turbo_ber_b128=Float64[], turbo_psr_b128=Float64[],
        sigma2_final=Float64[]
    )

    println("==============================================================")
    @printf("RSC replay-swap TURBO | eligible=%d using=%d | corr_thr=%.2f\n",
            length(eligible), length(blk_list), corr_thr)
    @printf("Turbo: σ2_init=%.3f eq_σ2_iters=%d turbo_iters=%d\n",
            σ2_init_rsc, eq_σ2_iters, turbo_iters)
    println("==============================================================")

    for (ii, b) in enumerate(blk_list)
        y = ComplexF64.(vec(ymat[b, :]))
        u_true = Int.(vec(umat[b, :]))
        b_true = Int.(vec(bmat[b, :]))

        hfull = ComplexF64.(vec(hmat[b, :]))
        Lh = min(length(hfull), length(y))
        h_use = hfull[1:Lh]

        u_hat, L01, σ2_fin = decode_rsc_turbo_eq_bcjr(
            y, h_use;
            σ2_init=σ2_init_rsc,
            eq_σ2_iters=eq_σ2_iters,
            turbo_iters=turbo_iters,
            llr_clip=llr_clip_rsc
        )
        b_hat = hard_from_llr(L01)

        push!(df_rsc, (
            blk=b,
            corr=Float64(corr[b]),
            turbo_ber_u64=ber(u_hat, u_true),
            turbo_psr_u64=psr_pkt(u_hat, u_true),
            turbo_ber_b128=ber(b_hat, b_true),
            turbo_psr_b128=psr_pkt(b_hat, b_true),
            sigma2_final=Float64(σ2_fin)
        ))

        if ii == 1 || ii % 25 == 0 || ii == length(blk_list)
            @printf("  RSC %4d/%d | corr=%.3f | u64 PSR=%.2f BER=%.3f | b128 BER=%.3f\n",
                    ii, length(blk_list), corr[b],
                    df_rsc.turbo_psr_u64[end], df_rsc.turbo_ber_u64[end], df_rsc.turbo_ber_b128[end])
        end
    end

    CSV.write(out_rsc, df_rsc)
    println("Saved → $out_rsc")
    @printf("RSC mean: u64 BER=%.4f PSR=%.4f | b128 BER=%.4f PSR=%.4f\n\n",
            mean(df_rsc.turbo_ber_u64), mean(df_rsc.turbo_psr_u64),
            mean(df_rsc.turbo_ber_b128), mean(df_rsc.turbo_psr_b128))

    # ============================================================
    # Part 2: RAW donor LDPC decode with EQ+SPA and JSDC
    # ============================================================
    isfile(raw_path)   || error("Missing raw file: $raw_path")
    isfile(cache_path) || error("Missing cache file: $cache_path (run scripts/build_ls_cache_bestD.jl first)")

    d = JLD2.load(raw_path)
    haskey(d, "all_packets_df") || error("Expected key all_packets_df in $raw_path")
    all_packets_df = DataFrame(d["all_packets_df"])
    packet_matrix = Matrix(select(all_packets_df, Not(:frame)))
    num_frames = size(packet_matrix, 1)

    dc = JLD2.load(cache_path)
    bestD = Vector{Int}(dc["bestD"])
    length(bestD) == num_frames || error("cache frames=$(length(bestD)) != rec frames=$num_frames")

    # donor LDPC(64->128)
    codeB, colsB, idrowsB, _ = initcode(64, 128, npc)
    codeB.icols === nothing && (encode(codeB, zeros(Int, 64)); nothing)
    HB = get_H_sparse(codeB)
    parityB = build_parity_indices(HB)

    # rebuild ground truth cw and x
    x_old_mat, cw_true_mat = rebuild_x_old_and_cw_mat(codeB; T_frame=T_frame, num_data=num_data, num_repeats=num_repeats)
    size(x_old_mat, 1) == num_frames || error("x_old_mat rows=$(size(x_old_mat,1)) != num_frames=$num_frames (fix num_data/num_repeats)")
    size(cw_true_mat, 1) == num_frames || error("cw_true_mat rows mismatch")

    useN = (nframes_use < 0) ? num_frames : min(nframes_use, num_frames)

    df_raw = DataFrame(
        frame=Int[],
        bestD=Int[],
        sigma2=Float64[],
        ber_eqspa=Float64[],
        ber_jsdc=Float64[],
        spa_iters=Int[]
    )

    println("==============================================================")
    @printf("RAW donor LDPC decode | frames=%d use=%d | LDPC(64->128) BPSK\n", num_frames, useN)
    println("==============================================================")

    for f in 1:useN
        y = extract_symbol_rate(packet_matrix[f, :], T_frame)
        y = shift_left(y, bestD[f])

        x_true  = vec(x_old_mat[f, :])
        cw_true = Int.(vec(cw_true_mat[f, :]))  # length 128

        # estimate h using known x_true
        h = ridge_ls_h(x_true, y, h_len, rho_ls)

        # noise variance estimate
        yhat = conv_prefix(h, x_true, T_frame)
        σ2 = Float64(mean(abs2, y .- yhat))

        # EQ+SPA
        Hc = convmtx_prefix(h, T_frame)
        A = Hc' * Hc
        @inbounds for ii in 1:T_frame
            A[ii,ii] += max(σ2, 1e-6)
        end
        rhs = Hc' * y
        x_lmmse = A \ rhs
        x_eq = real.(x_lmmse)
        cw_hat_eq, it_spa = spa_from_soft(codeB, colsB, parityB, x_eq, σ2; max_iter=50)
        ber_eqspa = mean(cw_hat_eq .!= cw_true)

        # JSDC (BPSK mode)
        pilot_pos = Int[]
        pilot_bpsk = Float64[]
        h_pos = collect(1:h_len)

        # warm start from 1-tap MF
        h1 = h[1]
        x_mf = real.(conj(h1) .* y) ./ max(abs2(h1), 1e-12)
        m_init = clamp.(x_mf, -0.999, 0.999)
        z_init = atanh.(m_init)
        L_prior = 2.0 .* z_init

        xhat_jsdc, _hhat_jsdc, _info = jsdc_qpsk_manual(
            y, codeB, parityB, pilot_pos, pilot_bpsk, h_pos;
            modulation=:bpsk,
            λ_par=jsdc_λpar,
            λ_pil=0.0,
            γ_z=jsdc_γz,
            γ_h=jsdc_γh,
            η_z=jsdc_ηz,
            η_h=0.0,
            max_iter=jsdc_max_iter,
            h_init=h[1:h_len],
            z_init=z_init,
            L_prior=L_prior,
            λ_prior=jsdc_λprior,
            σ2_data=max(σ2, 1e-6),
            verbose=false
        )
        ber_jsdc = mean(Int.(xhat_jsdc) .!= cw_true)

        push!(df_raw, (
            frame=f,
            bestD=bestD[f],
            sigma2=σ2,
            ber_eqspa=Float64(ber_eqspa),
            ber_jsdc=Float64(ber_jsdc),
            spa_iters=it_spa
        ))

        if f == 1 || f % 25 == 0 || f == useN
            @printf("  RAW %4d/%d | BER: EQ+SPA=%.3f JSDC=%.3f | σ2=%.3e\n",
                    f, useN, ber_eqspa, ber_jsdc, σ2)
        end
    end

    CSV.write(out_raw, df_raw)
    println("Saved → $out_raw")
    @printf("RAW mean: cw128 BER EQ+SPA=%.4f | JSDC=%.4f\n",
            mean(df_raw.ber_eqspa), mean(df_raw.ber_jsdc))

    println("\nDone. Two datasets, two CSVs. No background magic. 🪄")
end

main()
