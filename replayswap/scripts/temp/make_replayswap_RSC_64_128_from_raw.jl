#!/usr/bin/env julia
# scripts/make_replayswap_RSC_64_128_from_raw.jl
#
# Build a replay-swap base dataset for RSC(64->128) BPSK, directly into the
# recorded donor frames (logged_packets_and_ytrain.jld2).
#
# Output (data/):
#   replayswap_bpsk_RSC_64_128_from_realdata_donorLS_h20_rho1e-2.jld2
#
# Keys saved:
#   y_bpsk_swapped :: (Nblk × 128) ComplexF64 (imag=0)
#   u64_mat        :: (Nblk × 64)  Int
#   b128_mat       :: (Nblk × 128) Int   (coded bits)
#   frame_ids      :: (Nblk × 1)   Int   (which raw frame)
#   h_blk_mat      :: (Nblk × h_len) ComplexF64
#   corr_donor     :: (Nblk) Float64
#   meta_out       :: NamedTuple
#
# Run:
#   julia --project=. scripts/make_replayswap_RSC_64_128_from_raw.jl
#
# Options:
#   --rec data/raw/logged_packets_and_ytrain.jld2
#   --cache data/ls_cache_h20_rho1e-02_bestD.jld2
#   --outdir data
#   --T_frame 128
#   --h_len 20 --rho 1e-2 --Dmax 25
#   --num_data 20 --num_repeats 45
#   --seed 0x64128

using Random, Printf, Statistics, LinearAlgebra
using JLD2, DataFrames
using DSP
using SignalAnalysis

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
const LS = ensure_linksim_loaded!()
using .LS: initcode, encode, Code

# ----------------------------
# Helpers
# ----------------------------

# Extract symbol-rate samples from a recorded row: row[2:T_frame+1]
function extract_symbol_rate(row::AbstractVector{T}, T_frame::Int) where {T<:Number}
    @assert length(row) ≥ T_frame + 1
    return ComplexF64.(row[2:T_frame+1])
end

# shift y left by D (drop first D, pad zeros at end)
function shift_left(y::Vector{ComplexF64}, D::Int)
    T = length(y)
    D <= 0 && return y
    D >= T && return zeros(ComplexF64, T)
    return vcat(y[D+1:end], zeros(ComplexF64, D))
end

# Prefix convolution: y[t] = sum_{ℓ<=t} h[ℓ]*x[t-ℓ+1]
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

# y_new = y_obs + (h*x_new)[1:T] - (h*x_old)[1:T]
function replay_swap_prefix(y_obs::Vector{ComplexF64},
                            x_old::Vector{ComplexF64},
                            x_new::Vector{ComplexF64},
                            h::Vector{ComplexF64})
    T = length(y_obs)
    @assert length(x_old) == T
    @assert length(x_new) == T
    y_old_clean = conv_prefix(h, x_old, T)
    y_new_clean = conv_prefix(h, x_new, T)
    return y_obs .+ (y_new_clean .- y_old_clean)
end

# Ridge LS h on prefix model y ≈ X h
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

function corr_score(y::Vector{ComplexF64}, yhat::Vector{ComplexF64})
    num = abs(mean(y .* conj.(yhat)))
    den = sqrt(mean(abs2, y) * mean(abs2, yhat)) + 1e-12
    return num / den
end

# ----------------------------
# Donor x_old reconstruction (matches capture)
# ----------------------------
modulate_bpsk_bit(x::Int) = x == 1 ? (1.0 + 0im) : (-1.0 + 0im)

function rebuild_x_old_mat(codeB::Code; T_frame::Int, num_data::Int, num_repeats::Int)
    x_datas = zeros(ComplexF64, num_data, T_frame)
    for i in 1:num_data
        bseq   = mseq(11)[i : (codeB.k + i - 1)]
        d_test = Int.((bseq .+ 1) ./ 2)
        cw     = encode(codeB, d_test)
        bits   = Int.(cw)
        @inbounds for t in 1:T_frame
            x_datas[i, t] = modulate_bpsk_bit(bits[t])
        end
    end
    return repeat(x_datas, num_repeats, 1)
end

# ----------------------------
# LS cache build/load (bestD + h_hat)
# ----------------------------
function build_ls_cache_bestD!(rec_path::String, out_path::String;
                              h_len::Int, rho_ls::Float64, Dmax::Int,
                              T_frame::Int, npc::Int, num_data::Int, num_repeats::Int)

    @info "Building LS cache bestD" out_path
    d = JLD2.load(rec_path)
    haskey(d, "all_packets_df") || error("Expected key all_packets_df in $rec_path")

    all_packets_df = DataFrame(d["all_packets_df"])
    packet_matrix = Matrix(select(all_packets_df, Not(:frame)))
    nframes = size(packet_matrix, 1)

    codeB, _, _, _ = initcode(64, 128, npc)
    codeB.icols === nothing && (encode(codeB, zeros(Int, 64)); nothing)
    x_old_mat = rebuild_x_old_mat(codeB; T_frame=T_frame, num_data=num_data, num_repeats=num_repeats)
    size(x_old_mat, 1) == nframes || error("x_old_mat rows=$(size(x_old_mat,1)) != num_frames=$nframes")

    bestD = zeros(Int, nframes)
    corr_best = fill(-Inf, nframes)
    h_hat = zeros(ComplexF64, nframes, h_len)

    for f in 1:nframes
        y_frame = extract_symbol_rate(packet_matrix[f, :], T_frame)
        x_frame = ComplexF64.(vec(x_old_mat[f, :]))

        best_sc = -Inf
        best_d = 0
        best_h = zeros(ComplexF64, h_len)

        for D in 0:Dmax
            yD = shift_left(y_frame, D)
            h = ridge_ls_h(x_frame, yD, h_len, rho_ls)
            yhat = conv_prefix(h, x_frame, T_frame)
            sc = corr_score(yD, yhat)
            if sc > best_sc
                best_sc = sc
                best_d = D
                best_h = h
            end
        end

        bestD[f] = best_d
        corr_best[f] = best_sc
        h_hat[f, :] .= best_h

        if f == 1 || f % 200 == 0 || f == nframes
            @printf("cache frame %4d/%d | bestD=%2d | corr=%.4f\n", f, nframes, best_d, best_sc)
        end
    end

    meta = (
        note = "DFEC-built LS cache: per-frame ridge LS h_hat and bestD scan",
        h_len = h_len, rho_ls = rho_ls, Dmax = Dmax,
        T_frame = T_frame, num_frames = nframes,
        npc = npc, num_data = num_data, num_repeats = num_repeats,
        rec_path = rec_path,
    )

    @save out_path bestD corr_best h_hat meta
    return (bestD=bestD, corr_best=corr_best, h_hat=h_hat, meta=meta)
end

function load_or_build_ls_cache(rec_path::String, cache_path::String;
                               h_len::Int, rho_ls::Float64, Dmax::Int,
                               T_frame::Int, npc::Int, num_data::Int, num_repeats::Int)
    if isfile(cache_path)
        d = JLD2.load(cache_path)
        return (bestD=d["bestD"], corr_best=d["corr_best"], h_hat=d["h_hat"], meta=d["meta"])
    else
        mkpath(dirname(cache_path))
        return build_ls_cache_bestD!(rec_path, cache_path;
                                     h_len=h_len, rho_ls=rho_ls, Dmax=Dmax,
                                     T_frame=T_frame, npc=npc,
                                     num_data=num_data, num_repeats=num_repeats)
    end
end

# ----------------------------
# RSC(64->128) builder (rate-1/2 systematic)
# ----------------------------
const RSC_K  = 3
const RSC_FB = 0o7
const RSC_P  = 0o5

@inline function poly_bits(oct::Integer, K::Int)
    o = Int(oct)
    b = zeros(Int, K)
    @inbounds for i in 1:K
        b[i] = (o >> (i-1)) & 0x1
    end
    return b
end
const FB_BITS = poly_bits(RSC_FB, RSC_K)
const P_BITS  = poly_bits(RSC_P,  RSC_K)

function rsc_encode_parity(u::Vector{Int}, fb_bits::Vector{Int}, p_bits::Vector{Int}; K::Int=3)
    m = K - 1
    state = zeros(Int, m)
    N = length(u)
    p = Vector{Int}(undef, N)

    @inbounds for t in 1:N
        acc = 0
        for j in 1:m
            fb_bits[j+1] == 1 && (acc ⊻= state[j])
        end
        urec = u[t] ⊻ acc

        par = (p_bits[1] == 1) ? urec : 0
        for j in 1:m
            p_bits[j+1] == 1 && (par ⊻= state[j])
        end
        p[t] = par

        if m > 0
            for j in m:-1:2
                state[j] = state[j-1]
            end
            state[1] = urec
        end
    end
    return p
end

function pack_syspar(u::Vector{Int}, p::Vector{Int})
    N = length(u)
    @assert length(p) == N
    out = Vector{Int}(undef, 2N)
    @inbounds for t in 1:N
        out[2t-1] = u[t]
        out[2t]   = p[t]
    end
    return out
end

# BPSK mapping that matches your donor scripts: bit1->+1, bit0->-1
@inline bpsk_sym(b::Int) = (b == 1 ? 1.0 : -1.0) + 0im

function build_rsc_64_128_bpsk(rng::AbstractRNG)
    u64 = rand(rng, 0:1, 64)
    p64 = rsc_encode_parity(u64, FB_BITS, P_BITS; K=RSC_K)
    b128 = pack_syspar(u64, p64)                 # length 128
    x128 = ComplexF64[bpsk_sym(b128[i]) for i in 1:128]
    return x128, u64, b128
end

# ----------------------------
# Main
# ----------------------------
function main()
    rec_path   = joinpath(DATA_DIR, "raw", "logged_packets_and_ytrain.jld2")
    outdir     = DATA_DIR
    cache_path = joinpath(DATA_DIR, "ls_cache_h20_rho1e-02_bestD.jld2")

    # geometry (per-frame)
    T_frame = 128
    frames_per_block = 1
    T_block = T_frame

    # cache params
    h_len = 20
    rho_ls = 1e-2
    Dmax = 25

    # capture generation params for reconstructing donor x_old
    npc = 4
    num_data = 20
    num_repeats = 45

    seed = 0x64128

    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--rec"; i+=1; rec_path = ARGS[i]
        elseif a == "--outdir"; i+=1; outdir = ARGS[i]
        elseif a == "--cache"; i+=1; cache_path = ARGS[i]
        elseif a == "--T_frame"; i+=1; T_frame = parse(Int, ARGS[i])
        elseif a == "--h_len"; i+=1; h_len = parse(Int, ARGS[i])
        elseif a == "--rho"; i+=1; rho_ls = parse(Float64, ARGS[i])
        elseif a == "--Dmax"; i+=1; Dmax = parse(Int, ARGS[i])
        elseif a == "--num_data"; i+=1; num_data = parse(Int, ARGS[i])
        elseif a == "--num_repeats"; i+=1; num_repeats = parse(Int, ARGS[i])
        elseif a == "--seed"; i+=1; seed = parse(Int, ARGS[i])
        elseif a == "--help" || a == "-h"
            println("""
Usage:
  julia --project=. scripts/make_replayswap_RSC_64_128_from_raw.jl [args]

Paths:
  --rec    data/raw/logged_packets_and_ytrain.jld2
  --cache  data/ls_cache_h20_rho1e-02_bestD.jld2
  --outdir data

Geom:
  --T_frame 128   (fixed; per-frame replay-swap)

Cache:
  --h_len 20
  --rho   1e-2
  --Dmax  25

Donor recon:
  --num_data 20
  --num_repeats 45

Other:
  --seed 0x64128
""")
            return
        else
            error("Unknown arg: $a")
        end
        i += 1
    end
    T_block = T_frame

    isfile(rec_path) || error("Missing rec file: $rec_path")
    mkpath(outdir)

    # Load recorded packets
    drec = JLD2.load(rec_path)
    haskey(drec, "all_packets_df") || error("Expected key all_packets_df in $rec_path")
    all_packets_df = DataFrame(drec["all_packets_df"])
    packet_matrix = Matrix(select(all_packets_df, Not(:frame)))
    num_frames = size(packet_matrix, 1)

    # Load/build cache
    cache = load_or_build_ls_cache(rec_path, cache_path;
                                   h_len=h_len, rho_ls=rho_ls, Dmax=Dmax,
                                   T_frame=T_frame, npc=npc,
                                   num_data=num_data, num_repeats=num_repeats)
    bestD = cache.bestD

    # Rebuild donor x_old per frame (LDPC64->128 BPSK, matches capture)
    codeB, _, _, _ = initcode(64, 128, npc)
    codeB.icols === nothing && (encode(codeB, zeros(Int, 64)); nothing)
    x_old_mat = rebuild_x_old_mat(codeB; T_frame=T_frame, num_data=num_data, num_repeats=num_repeats)
    size(x_old_mat, 1) == num_frames || error("x_old_mat rows mismatch")

    # Each frame becomes one block
    Nblk = num_frames
    rng = MersenneTwister(seed)

    y_rsc = Matrix{ComplexF64}(undef, Nblk, T_block)
    u64_mat = Matrix{Int}(undef, Nblk, 64)
    b128_mat = Matrix{Int}(undef, Nblk, 128)
    h_blk_mat = Matrix{ComplexF64}(undef, Nblk, h_len)
    corr_donor = Vector{Float64}(undef, Nblk)
    frame_ids = reshape(collect(1:Nblk), Nblk, 1)

    println("==============================================================")
    @printf("Replay-swap RSC(64->128) BPSK | frames=%d | T_frame=%d | h_len=%d rho=%.2e Dmax=%d\n",
            num_frames, T_frame, h_len, rho_ls, Dmax)
    println("==============================================================")

    for f in 1:num_frames
        y_f = extract_symbol_rate(packet_matrix[f, :], T_frame)
        y_f = shift_left(y_f, bestD[f])

        x_old = ComplexF64.(vec(x_old_mat[f, :]))
        @assert length(x_old) == T_block
        @assert length(y_f) == T_block

        # block channel from donor
        h_blk = ridge_ls_h(x_old, y_f, h_len, rho_ls)
        yhat_old = conv_prefix(h_blk, x_old, T_block)
        corr = corr_score(y_f, yhat_old)

        # build new RSC(64->128) codeword -> BPSK symbols
        x_new, u64, b128 = build_rsc_64_128_bpsk(rng)

        # replay swap
        y_new = replay_swap_prefix(y_f, x_old, x_new, h_blk)

        y_rsc[f, :] .= y_new
        u64_mat[f, :] .= u64
        b128_mat[f, :] .= b128
        h_blk_mat[f, :] .= h_blk
        corr_donor[f] = corr

        if f == 1 || f % 200 == 0 || f == num_frames
            @printf("frame %4d/%d | corr_donor=%.4f | bestD=%d\n", f, num_frames, corr, bestD[f])
        end
    end

    out_rsc = joinpath(outdir, "replayswap_bpsk_RSC_64_128_from_realdata_donorLS_h20_rho1e-2.jld2")
    meta_out = (
        note = "DFEC replay-swap from raw donor capture -> RSC(64->128) BPSK target",
        conv_geometry = "prefix: y_clean = conv(h, x)[1:T_frame]",
        T_frame = T_frame,
        frames_per_block = 1,
        blocks = Nblk,
        h_len = h_len,
        rho_blk = rho_ls,
        cache_path = cache_path,
        donor_code = "LDPC(64->128) used only to reconstruct x_old (must match capture)",
        rsc = (K=RSC_K, FB=RSC_FB, P=RSC_P),
        bpsk_mapping = "bit1->+1, bit0->-1",
        seed = seed
    )

    @save out_rsc y_bpsk_swapped=y_rsc u64_mat b128_mat frame_ids h_blk_mat corr_donor meta_out
    println("\nSaved → $out_rsc")
end

main()
