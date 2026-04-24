#!/usr/bin/env julia
# scripts/build_ls_cache_bestD.jl
#
# Build ls_cache_h20_rho1e-02_bestD.jld2 from logged_packets_and_ytrain.jld2
#
# Output file contains:
#   bestD      :: Vector{Int}          (per-frame best delay)
#   corr_best  :: Vector{Float64}      (per-frame best corr score)
#   h_hat      :: Matrix{ComplexF64}   (num_frames × h_len) ridge-LS channel
#   meta       :: NamedTuple           (h_len, rho_ls, Dmax, etc.)

using Random, Printf, Statistics, LinearAlgebra
using JLD2, DataFrames
using SignalAnalysis
using DSP

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
const LS = ensure_linksim_loaded!()
using .LS: initcode, encode, Code

# ----------------------------
# Config defaults (match your naming)
# ----------------------------
const npc_default = 4
const T_frame_default = 128

# capture rebuild params (must match how logged_packets were generated)
const num_train_default   = 5
const num_data_default    = 20
const gap_default         = 160
const num_repeats_default = 45

# ----------------------------
# Helpers
# ----------------------------
modulate_bpsk_bit(x::Int) = x == 1 ? (1.0 + 0im) : (-1.0 + 0im)

function invperm_vec(π::Vector{Int})
    πinv = similar(π)
    @inbounds for i in 1:length(π)
        πinv[π[i]] = i
    end
    return πinv
end

function extract_symbol_rate(row::AbstractVector{T}, T_frame::Int) where {T<:Number}
    @assert length(row) ≥ T_frame + 1
    if T <: Complex
        return row[2:T_frame+1]
    else
        return ComplexF64.(row[2:T_frame+1])
    end
end

# Prefix conv: yhat[t] = sum_{ℓ<=t} h[ℓ]*x[t-ℓ+1]
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

function convmtx_prefix(x::Vector{ComplexF64}, h_len::Int)
    T = length(x)
    X = zeros(ComplexF64, T, h_len)
    @inbounds for t in 1:T
        for k in 1:h_len
            idx = t - k + 1
            X[t, k] = idx >= 1 ? x[idx] : (0.0 + 0im)
        end
    end
    return X
end

function ridge_ls_h(x::Vector{ComplexF64}, y::Vector{ComplexF64}, h_len::Int, rho::Float64)
    X = convmtx_prefix(x, h_len)
    A = X'X
    @inbounds for i in 1:h_len
        A[i,i] += rho
    end
    b = X'y
    return Vector{ComplexF64}(A \ b)
end

# normalized correlation score (bigger is better)
function corr_score(y::Vector{ComplexF64}, yhat::Vector{ComplexF64})
    num = abs(mean(y .* conj.(yhat)))
    den = sqrt(mean(abs2, y) * mean(abs2, yhat)) + 1e-12
    return num / den
end

# shift y left by D (drop first D, pad zeros at end)
function shift_left(y::Vector{ComplexF64}, D::Int)
    T = length(y)
    D <= 0 && return y
    D >= T && return zeros(ComplexF64, T)
    return vcat(y[D+1:end], zeros(ComplexF64, D))
end

"""
Rebuild donor x_old frames (num_frames × T_frame) the same way as capture:
- codeB is (64→128) LDPC
- For each "data frame" i, use mseq(11) slice to make 64 info bits, encode to 128, modulate to ±1
- Repeat the 20 unique frames num_repeats times
"""
function rebuild_x_old_mat(codeB::Code;
                           T_frame::Int,
                           num_train::Int,
                           num_data::Int,
                           gap::Int,
                           num_repeats::Int)

    # Build the 20 unique codewords/symbol frames
    x_datas = zeros(ComplexF64, num_data, T_frame)

    for i in 1:num_data
        bseq   = mseq(11)[i : (codeB.k + i - 1)]
        d_test = Int.((bseq .+ 1) ./ 2)          # map {-1,+1} -> {0,1}
        cw     = encode(codeB, d_test)           # BitVector
        bits   = Int.(cw)                        # 0/1
        @inbounds for t in 1:T_frame
            x_datas[i, t] = modulate_bpsk_bit(bits[t])
        end
    end

    # Repeat to match recorded length
    return repeat(x_datas, num_repeats, 1)  # (num_data*num_repeats) × T_frame
end

# ----------------------------
# Main
# ----------------------------
function main()
    # defaults
    rec_path   = joinpath(DATA_DIR, "logged_packets_and_ytrain.jld2")
    out_path   = joinpath(DATA_DIR, "ls_cache_h20_rho1e-02_bestD.jld2")

    h_len   = 20
    rho_ls  = 1e-2
    Dmax    = 25

    npc     = npc_default
    T_frame = T_frame_default
    num_train   = num_train_default
    num_data    = num_data_default
    gap         = gap_default
    num_repeats = num_repeats_default

    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--rec"; i+=1; rec_path = ARGS[i]
        elseif a == "--out"; i+=1; out_path = ARGS[i]
        elseif a == "--h_len"; i+=1; h_len = parse(Int, ARGS[i])
        elseif a == "--rho"; i+=1; rho_ls = parse(Float64, ARGS[i])
        elseif a == "--Dmax"; i+=1; Dmax = parse(Int, ARGS[i])
        elseif a == "--T_frame"; i+=1; T_frame = parse(Int, ARGS[i])
        elseif a == "--num_data"; i+=1; num_data = parse(Int, ARGS[i])
        elseif a == "--num_repeats"; i+=1; num_repeats = parse(Int, ARGS[i])
        elseif a == "--help" || a == "-h"
            println("""
Usage:
  julia --project=. scripts/build_ls_cache_bestD.jl [args]

Paths:
  --rec  data/logged_packets_and_ytrain.jld2
  --out  data/ls_cache_h20_rho1e-02_bestD.jld2

LS params:
  --h_len 20
  --rho   1e-2
  --Dmax  25

Donor frame geom:
  --T_frame 128
  --num_data 20
  --num_repeats 45
""")
            return
        else
            error("Unknown arg: $a")
        end
        i += 1
    end

    isfile(rec_path) || error("Missing rec file: $rec_path")
    mkpath(dirname(out_path))

    # Donor code: 64->128 (matches your capture)
    codeB, _, _, _ = initcode(64, 128, npc)
    codeB.icols === nothing && (encode(codeB, zeros(Int, 64)); nothing)

    # Rebuild x_old per frame
    x_old_mat = rebuild_x_old_mat(codeB; T_frame=T_frame, num_train=num_train,
                                  num_data=num_data, gap=gap, num_repeats=num_repeats)

    # Load recorded packets
    @info "Loading recorded data" rec_path
    d = JLD2.load(rec_path)
    haskey(d, "all_packets_df") || error("Expected key all_packets_df in $rec_path")
    all_packets_df = DataFrame(d["all_packets_df"])
    packet_matrix = Matrix(select(all_packets_df, Not(:frame)))

    num_frames = size(packet_matrix, 1)
    @info "Frames" num_frames
    size(x_old_mat, 1) == num_frames || error("x_old_mat rows=$(size(x_old_mat,1)) != num_frames=$num_frames. Adjust --num_data/--num_repeats.")

    bestD = zeros(Int, num_frames)
    corr_best = fill(-Inf, num_frames)
    h_hat = zeros(ComplexF64, num_frames, h_len)

    println("==============================================================")
    @printf("BUILD LS CACHE: h_len=%d rho=%.2e Dmax=%d | frames=%d\n", h_len, rho_ls, Dmax, num_frames)
    println("==============================================================")

    # Per-frame scan
    for f in 1:num_frames
        y_frame = ComplexF64.(collect(extract_symbol_rate(packet_matrix[f, :], T_frame)))
        x_frame = ComplexF64.(vec(x_old_mat[f, :]))

        best_sc = -Inf
        best_d  = 0
        best_h  = zeros(ComplexF64, h_len)

        for D in 0:Dmax
            yD = shift_left(y_frame, D)
            h = ridge_ls_h(x_frame, yD, h_len, rho_ls)
            yhat = conv_prefix(h, x_frame, T_frame)
            sc = corr_score(yD, yhat)
            if sc > best_sc
                best_sc = sc
                best_d  = D
                best_h  = h
            end
        end

        bestD[f] = best_d
        corr_best[f] = best_sc
        h_hat[f, :] .= best_h

        if f == 1 || f % 100 == 0 || f == num_frames
            @printf("frame %4d/%d | bestD=%2d | corr=%.4f\n", f, num_frames, best_d, best_sc)
        end
    end

    meta = (
        note = "DFEC-built LS cache: per-frame ridge LS h_hat and bestD scan",
        h_len = h_len,
        rho_ls = rho_ls,
        Dmax = Dmax,
        T_frame = T_frame,
        num_frames = num_frames,
        npc = npc,
        num_data = num_data,
        num_repeats = num_repeats,
        rec_path = rec_path,
    )

    @info "Saving cache" out_path
    @save out_path bestD corr_best h_hat meta
    println("Saved → $out_path")
end

main()
