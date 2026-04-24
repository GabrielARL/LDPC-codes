#!/usr/bin/env julia
# scripts/sweep_raw_jsdc_hypers.jl
#
# Sweep JSDC hyperparameters on the LITERAL donor capture frames
# (no replay-swap, no augmentation): LDPC(64->128) BPSK.
#
# Uses:
#   - alignment bestD from ls_cache_h20_rho1e-02_bestD.jld2
#   - per-frame ridge-LS channel h (using TRUE x_old)
#   - MF warm start for z_init, and turbo-style prior from MF
#   - σ2_data for data-term normalization inside jsdc_qpsk_manual (requires patched core)
#
# Outputs:
#   CSV with mean BER across selected frames (and optional "hard" frames by σ² threshold)
#
# Example:
#   julia --project=. scripts/sweep_raw_jsdc_hypers.jl \
#     --nframes 200 --seed_sel 123 \
#     --etaz "3e-4,1e-3" --lampar "0.05,0.1,0.2" --lamprior "0.5,1.0,2.0" \
#     --maxiter "100,200" --outcsv data/runs/sweep_raw_jsdc.csv
#
# Notes:
# - SPA/EQ baselines are computed once for reference, but the sweep is over JSDC only.
# - Your donor mapping is bit1->+1, bit0->-1. SPA convention differs; handled internally.

using Random, Printf, Statistics, LinearAlgebra
using SparseArrays
using JLD2, DataFrames, CSV
using DSP
using SignalAnalysis

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
const LS = ensure_linksim_loaded!()
using .LS: initcode, encode, get_H_sparse, sum_product_decode, jsdc_qpsk_manual, Code

# ----------------------------
# Helpers (match your pipeline)
# ----------------------------

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

function rebuild_x_old_and_cw_mat(codeB::Code; T_frame::Int, num_data::Int, num_repeats::Int)
    x_mat  = zeros(ComplexF64, num_data, T_frame)
    cw_mat = zeros(Int,        num_data, T_frame)

    for i in 1:num_data
        bseq   = mseq(11)[i : (codeB.k + i - 1)]
        d_test = Int.((bseq .+ 1) ./ 2)  # {-1,+1} -> {0,1}
        cw     = encode(codeB, d_test)
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

# SPA wrapper:
# sum_product_decode decides bit=1 when L_post<0, which corresponds to bit1 -> negative "y".
# Our donor mapping is bit1 -> +1, so we negate x_soft before SPA.
@inline function spa_from_soft(codeB::Code,
                               colsB::Vector{Vector{Int}},
                               parityB::Vector{Vector{Int}},
                               x_soft::Vector{Float64},
                               σ2::Float64;
                               max_iter::Int=50)
    H = get_H_sparse(codeB)
    y_for_spa = -x_soft
    x_hat, iters = sum_product_decode(H, y_for_spa, max(σ2, 1e-6), parityB, colsB; max_iter=max_iter)
    return x_hat, iters
end

# ----------------------------
# CLI parsing
# ----------------------------

function parse_list_float(s::String)
    ss = replace(strip(s), " " => "")
    isempty(ss) && return Float64[]
    return Float64.(parse.(Float64, split(ss, ",")))
end

function parse_list_int(s::String)
    ss = replace(strip(s), " " => "")
    isempty(ss) && return Int[]
    return Int.(parse.(Int, split(ss, ",")))
end

# ----------------------------
# Main
# ----------------------------
function main()
    rec_path   = joinpath(DATA_DIR, "raw", "logged_packets_and_ytrain.jld2")
    cache_path = joinpath(DATA_DIR, "ls_cache_h20_rho1e-02_bestD.jld2")
    outcsv     = joinpath(DATA_DIR, "runs", "sweep_raw_jsdc.csv")

    # capture geometry (must match)
    npc = 4
    T_frame = 128
    num_data = 20
    num_repeats = 45

    # channel est
    h_len = 20
    rho_ls = 1e-2

    # selection
    nframes_use = 200
    seed_sel = 123

    # Sweep grids
    etaz_list     = [3e-4, 1e-3]
    lampar_list   = [0.05, 0.10, 0.20]
    lamprior_list = [0.5, 1.0, 2.0]
    maxiter_list  = [100, 200]

    # fixed JSDC reg terms
    γ_z = 5e-3
    γ_h = 1e-3

    # hard-frame reporting
    sigma2_hard = 0.5

    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--rec"; i+=1; rec_path = ARGS[i]
        elseif a == "--cache"; i+=1; cache_path = ARGS[i]
        elseif a == "--outcsv"; i+=1; outcsv = ARGS[i]
        elseif a == "--nframes"; i+=1; nframes_use = parse(Int, ARGS[i])
        elseif a == "--seed_sel"; i+=1; seed_sel = parse(Int, ARGS[i])
        elseif a == "--etaz"; i+=1; etaz_list = parse_list_float(ARGS[i])
        elseif a == "--lampar"; i+=1; lampar_list = parse_list_float(ARGS[i])
        elseif a == "--lamprior"; i+=1; lamprior_list = parse_list_float(ARGS[i])
        elseif a == "--maxiter"; i+=1; maxiter_list = parse_list_int(ARGS[i])
        elseif a == "--sigma2_hard"; i+=1; sigma2_hard = parse(Float64, ARGS[i])
        elseif a == "--help" || a == "-h"
            println("""
Usage:
  julia --project=. scripts/sweep_raw_jsdc_hypers.jl [args]

Paths:
  --rec    data/raw/logged_packets_and_ytrain.jld2
  --cache  data/ls_cache_h20_rho1e-02_bestD.jld2
  --outcsv data/runs/sweep_raw_jsdc.csv

Selection:
  --nframes 200
  --seed_sel 123

Sweep grids:
  --etaz     "3e-4,1e-3"
  --lampar   "0.05,0.1,0.2"
  --lamprior "0.5,1.0,2.0"
  --maxiter  "100,200"

Reporting:
  --sigma2_hard 0.5
""")
            return
        else
            error("Unknown arg: $a")
        end
        i += 1
    end

    isfile(rec_path) || error("Missing rec file: $rec_path")
    isfile(cache_path) || error("Missing cache file: $cache_path")
    mkpath(dirname(outcsv))

    # Load recorded packets
    d = JLD2.load(rec_path)
    haskey(d, "all_packets_df") || error("Expected key all_packets_df in $rec_path")
    all_packets_df = DataFrame(d["all_packets_df"])
    packet_matrix = Matrix(select(all_packets_df, Not(:frame)))
    num_frames = size(packet_matrix, 1)

    # Load LS cache
    dc = JLD2.load(cache_path)
    bestD = Vector{Int}(dc["bestD"])
    length(bestD) == num_frames || error("cache frames=$(length(bestD)) != rec frames=$num_frames")

    # LDPC code (64->128)
    codeB, colsB, idrowsB, _ = initcode(64, 128, npc)
    codeB.icols === nothing && (encode(codeB, zeros(Int, 64)); nothing)
    HB = get_H_sparse(codeB)
    parityB = build_parity_indices(HB)

    # Rebuild true donor codeword bits + transmitted symbols
    x_old_mat, cw_true_mat = rebuild_x_old_and_cw_mat(codeB; T_frame=T_frame, num_data=num_data, num_repeats=num_repeats)
    size(x_old_mat, 1) == num_frames || error("x_old_mat rows=$(size(x_old_mat,1)) != num_frames=$num_frames. Fix num_data/num_repeats.")
    size(cw_true_mat, 1) == num_frames || error("cw_true_mat rows mismatch")

    # Select frames
    frames = collect(1:num_frames)
    rng = MersenneTwister(seed_sel)
    shuffle!(rng, frames)
    useN = (nframes_use < 0) ? num_frames : min(nframes_use, num_frames)
    frames = frames[1:useN]

    println("==============================================================")
    @printf("SWEEP RAW JSDC | frames=%d (of %d) | hard σ²>%.3f\n", useN, num_frames, sigma2_hard)
    @printf("etaz=%s\n", string(etaz_list))
    @printf("lampar=%s\n", string(lampar_list))
    @printf("lamprior=%s\n", string(lamprior_list))
    @printf("maxiter=%s\n", string(maxiter_list))
    println("==============================================================")

    # Precompute per-frame inputs (so sweeps are faster / deterministic)
    y_list      = Vector{Vector{ComplexF64}}(undef, useN)
    h_list      = Vector{Vector{ComplexF64}}(undef, useN)
    sigma2_list = Vector{Float64}(undef, useN)
    cw_list     = Vector{Vector{Int}}(undef, useN)
    zinit_list  = Vector{Vector{Float64}}(undef, useN)
    Lprior_list = Vector{Vector{Float64}}(undef, useN)

    # Also compute baselines once (MF-SPA and EQ-SPA)
    ber_spa_base = Vector{Float64}(undef, useN)
    ber_eq_base  = Vector{Float64}(undef, useN)

    for (ii, f) in enumerate(frames)
        y = extract_symbol_rate(packet_matrix[f, :], T_frame)
        y = shift_left(y, bestD[f])

        x_true  = vec(x_old_mat[f, :])
        cw_true = Int.(vec(cw_true_mat[f, :]))

        # LS h
        h = ridge_ls_h(x_true, y, h_len, rho_ls)

        # σ² from residual using TRUE x_true
        yhat = conv_prefix(h, x_true, T_frame)
        e = y .- yhat
        σ2 = mean(abs2, e)

        # MF soft (bit1 -> +)
        h1 = h[1]
        x_mf = real.(conj(h1) .* y) ./ max(abs2(h1), 1e-12)

        # Warm start + prior
        m_init = clamp.(x_mf, -0.999, 0.999)
        z_init = atanh.(m_init)
        L_prior = 2.0 .* z_init   # so tanh(L/2) ≈ m_init

        # Baseline SPA
        xhat_spa, _ = spa_from_soft(codeB, colsB, parityB, x_mf, σ2; max_iter=50)
        ber_spa = mean(Int.(xhat_spa) .!= cw_true)

        # Baseline EQ (LMMSE deconv)
        Hc = convmtx_prefix(h, T_frame)
        A = Hc' * Hc
        @inbounds for k in 1:T_frame
            A[k,k] += σ2
        end
        b = Hc' * y
        x_lmmse = A \ b
        x_eq = real.(x_lmmse)

        xhat_eq, _ = spa_from_soft(codeB, colsB, parityB, x_eq, σ2; max_iter=50)
        ber_eq = mean(Int.(xhat_eq) .!= cw_true)

        y_list[ii]      = y
        h_list[ii]      = h[1:h_len]
        sigma2_list[ii] = σ2
        cw_list[ii]     = cw_true
        zinit_list[ii]  = z_init
        Lprior_list[ii] = L_prior
        ber_spa_base[ii] = ber_spa
        ber_eq_base[ii]  = ber_eq
    end

    @printf("Baseline mean BER: SPA=%.4f  EQ=%.4f\n", mean(ber_spa_base), mean(ber_eq_base))
    hard_mask = sigma2_list .> sigma2_hard
    if any(hard_mask)
        @printf("Baseline HARD mean BER (σ²>%.3f): SPA=%.4f  EQ=%.4f  (n=%d)\n",
                sigma2_hard, mean(ber_spa_base[hard_mask]), mean(ber_eq_base[hard_mask]), count(hard_mask))
    end

    # Sweep
    results = DataFrame(
        cfg=Int[],
        etaz=Float64[],
        lampar=Float64[],
        lamprior=Float64[],
        maxiter=Int[],
        mean_ber=Float64[],
        mean_ber_hard=Float64[],
        nframes=Int[],
        nhard=Int[],
        mean_sigma2=Float64[],
        mean_sigma2_hard=Float64[],
        base_spa=Float64[],
        base_eq=Float64[],
    )

    cfg = 0
    total_cfg = length(etaz_list) * length(lampar_list) * length(lamprior_list) * length(maxiter_list)
    for etaz in etaz_list, lampar in lampar_list, lamprior in lamprior_list, maxit in maxiter_list
        cfg += 1
        ber_jsdc = Vector{Float64}(undef, useN)

        pilot_pos = Int[]
        pilot_bpsk = Float64[]
        h_pos = collect(1:h_len)

        for ii in 1:useN
            xhat, _, _ = jsdc_qpsk_manual(
                y_list[ii], codeB, parityB, pilot_pos, pilot_bpsk, h_pos;
                modulation=:bpsk,
                λ_par=lampar,
                λ_pil=0.0,
                γ_z=γ_z,
                γ_h=γ_h,
                η_z=etaz,
                η_h=0.0,                      # freeze channel
                max_iter=maxit,
                h_init=h_list[ii],
                z_init=zinit_list[ii],
                L_prior=Lprior_list[ii],
                λ_prior=lamprior,
                σ2_data=sigma2_list[ii],       # requires patched core
                verbose=false
            )
            ber_jsdc[ii] = mean(Int.(xhat) .!= cw_list[ii])
        end

        mber = mean(ber_jsdc)
        mber_h = any(hard_mask) ? mean(ber_jsdc[hard_mask]) : NaN
        ms2 = mean(sigma2_list)
        ms2_h = any(hard_mask) ? mean(sigma2_list[hard_mask]) : NaN

        push!(results, (
            cfg=cfg,
            etaz=etaz,
            lampar=lampar,
            lamprior=lamprior,
            maxiter=maxit,
            mean_ber=mber,
            mean_ber_hard=mber_h,
            nframes=useN,
            nhard=count(hard_mask),
            mean_sigma2=ms2,
            mean_sigma2_hard=ms2_h,
            base_spa=mean(ber_spa_base),
            base_eq=mean(ber_eq_base),
        ))

        if cfg == 1 || cfg % 5 == 0 || cfg == total_cfg
            @printf("cfg %3d/%d | ηz=%g λpar=%g λprior=%g it=%d | BER=%.4f (hard=%.4f)\n",
                    cfg, total_cfg, etaz, lampar, lamprior, maxit, mber, mber_h)
        end
    end

    CSV.write(outcsv, results)
    println("Saved → $outcsv")

    # Print best few
    sort!(results, :mean_ber)
  println("\nTop 8 configs by mean_ber:")
topN = min(8, nrow(results))
for r in eachrow(first(results, topN))
    @printf("cfg=%3d | ηz=%g λpar=%g λprior=%g it=%d | BER=%.4f (hard=%.4f)\n",
            r.cfg, r.etaz, r.lampar, r.lamprior, r.maxiter, r.mean_ber, r.mean_ber_hard)
end
println()

end

main()
