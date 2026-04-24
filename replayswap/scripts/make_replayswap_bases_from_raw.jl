#!/usr/bin/env julia
# scripts/make_replayswap_bases_from_raw.jl
#
# DFEC self-contained builder:
#   raw capture (logged_packets_and_ytrain.jld2)
#   -> (optional) LS cache bestD/h_hat (ls_cache_h20_rho1e-02_bestD.jld2)
#   -> replay-swap base datasets:
#        (A) LDPC concat 256→512→1024 (QPSK)
#        (B) RSC  concat 256→512→1024 (QPSK)  [outer+inner RSC rate-1/2]
#
# Outputs (DFEC/data/):
#   replayswap_qpsk_concat_256_512_1024_from_realdata_donorLS_h20_rho1e-2.jld2
#   replayswap_qpsk_RSCconcat_256_512_1024_from_realdata_donorLS_h20_rho1e-2.jld2
#
# Run:
#   julia --project=. scripts/make_replayswap_bases_from_raw.jl
#
# Optional:
#   julia --project=. scripts/make_replayswap_bases_from_raw.jl \
#     --rec data/raw/logged_packets_and_ytrain.jld2 \
#     --outdir data \
#     --h_len 20 --rho 1e-2 --Dmax 25 \
#     --frames_per_block 4 --T_frame 128 \
#     --seed 0x2565121024

using Random, Printf, Statistics, LinearAlgebra
using JLD2, DataFrames
using DSP
using SignalAnalysis

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
include(joinpath(ROOT, "lib", "ModemQPSK.jl"))
using .ModemQPSK: qpsk_from_bits, bpsk_from_bits

const LS = ensure_linksim_loaded!()
using .LS: initcode, encode, Code

# ----------------------------
# Small helpers
# ----------------------------
@inline function invperm_vec(π::Vector{Int})
    πinv = similar(π)
    @inbounds for i in 1:length(π)
        πinv[π[i]] = i
    end
    return πinv
end

# Extract symbol-rate samples from a recorded row.
# Your original scripts used row[2:T_frame+1].
function extract_symbol_rate(row::AbstractVector{T}, T_frame::Int) where {T<:Number}
    @assert length(row) ≥ T_frame + 1
    if T <: Complex
        return ComplexF64.(row[2:T_frame+1])
    else
        return ComplexF64.(row[2:T_frame+1])
    end
end

# Shift-left helper (timing alignment): drop first D samples, pad zeros
function shift_left(y::Vector{ComplexF64}, D::Int)
    T = length(y)
    D <= 0 && return y
    D >= T && return zeros(ComplexF64, T)
    return vcat(y[D+1:end], zeros(ComplexF64, D))
end

# Prefix convolution: y[t] = sum_{ℓ<=t} h[ℓ] * x[t-ℓ+1]
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

# Ridge LS h on prefix conv model y ≈ X h
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
# Donor x_old reconstruction (must match capture)
# ----------------------------
modulate_bpsk_bit(x::Int) = x == 1 ? (1.0 + 0im) : (-1.0 + 0im)

function rebuild_x_old_mat(codeB::Code; T_frame::Int, num_data::Int, num_repeats::Int)
    # Use same pattern as your original: mseq(11) slices generate info bits
    x_datas = zeros(ComplexF64, num_data, T_frame)
    for i in 1:num_data
        bseq   = mseq(11)[i : (codeB.k + i - 1)]
        d_test = Int.((bseq .+ 1) ./ 2)       # {-1,+1} -> {0,1}
        cw     = encode(codeB, d_test)        # BitVector
        bits   = Int.(cw)
        @inbounds for t in 1:T_frame
            x_datas[i, t] = modulate_bpsk_bit(bits[t])
        end
    end
    return repeat(x_datas, num_repeats, 1)    # (num_data*num_repeats) × T_frame
end

# ----------------------------
# Build LS cache (bestD per frame) if missing
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

    size(x_old_mat, 1) == nframes || error("x_old_mat rows=$(size(x_old_mat,1)) != num_frames=$nframes. Adjust --num_data/--num_repeats")

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
        h_len = h_len,
        rho_ls = rho_ls,
        Dmax = Dmax,
        T_frame = T_frame,
        num_frames = nframes,
        npc = npc,
        num_data = num_data,
        num_repeats = num_repeats,
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
# LDPC concat target builder (256→512→1024)
# ----------------------------
function build_ldpc_concat_qpsk(rng::AbstractRNG, codeO::Code, codeI::Code, π512::Union{Nothing,Vector{Int}})
    u256 = rand(rng, 0:1, codeO.k)
    cw512 = encode(codeO, u256)
    b512 = Int.(cw512)
    b512_i = isnothing(π512) ? b512 : b512[π512]
    cw1024 = encode(codeI, b512_i)
    b1024 = Int.(cw1024)
    s512 = ComplexF64.(qpsk_from_bits(b1024))
    return s512, u256, b512, b512_i, b1024
end

# ----------------------------
# RSC concat target builder (256→512→1024) rate-1/2 systematic
# b512 is coded bits length 512 as [u1,p1,u2,p2,...]
# b1024 similarly.
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

# RSC encoder: input u (0/1), output parity p (0/1), with feedback
function rsc_encode_parity(u::Vector{Int}, fb_bits::Vector{Int}, p_bits::Vector{Int}; K::Int=3)
    m = K - 1
    state = zeros(Int, m) # shift register bits (most recent first)
    N = length(u)
    p = Vector{Int}(undef, N)

    @inbounds for t in 1:N
        # feedback: urec = u ⊻ (fb · state)
        acc = 0
        for j in 1:m
            fb_bits[j+1] == 1 && (acc ⊻= state[j])
        end
        urec = u[t] ⊻ acc

        # parity: p = (p_bits[1]*urec) ⊻ (p_bits[2:]*state)
        par = (p_bits[1] == 1) ? urec : 0
        for j in 1:m
            p_bits[j+1] == 1 && (par ⊻= state[j])
        end
        p[t] = par

        # shift in urec
        if m > 0
            for j in m:-1:2
                state[j] = state[j-1]
            end
            state[1] = urec
        end
    end

    return p
end

# pack systematic+parity into length 2N vector [u1,p1,u2,p2,...]
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

function build_rsc_concat_qpsk(rng::AbstractRNG, π512::Union{Nothing,Vector{Int}})
    # outer input u256
    u256 = rand(rng, 0:1, 256)
    p256 = rsc_encode_parity(u256, FB_BITS, P_BITS; K=RSC_K)
    b512 = pack_syspar(u256, p256)                 # length 512

    b512_i = isnothing(π512) ? b512 : b512[π512]   # interleaver on 512 bits

    p512 = rsc_encode_parity(b512_i, FB_BITS, P_BITS; K=RSC_K)
    b1024 = pack_syspar(b512_i, p512)              # length 1024

    s512 = ComplexF64.(qpsk_from_bits(b1024))
    return s512, u256, b512, b512_i, b1024
end

# ----------------------------
# Main
# ----------------------------
function main()
    rec_path   = joinpath(DATA_DIR, "raw", "logged_packets_and_ytrain.jld2")
    outdir     = DATA_DIR
    cache_path = joinpath(DATA_DIR, "ls_cache_h20_rho1e-02_bestD.jld2")

    # geometry
    T_frame = 128
    frames_per_block = 4
    T_block = T_frame * frames_per_block

    # cache params
    h_len = 20
    rho_ls = 1e-2
    Dmax = 25

    # capture generation params
    npc = 4
    num_data = 20
    num_repeats = 45

    seed = 0x2565121024
    use_interleaver = true

    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--rec"; i+=1; rec_path = ARGS[i]
        elseif a == "--outdir"; i+=1; outdir = ARGS[i]
        elseif a == "--cache"; i+=1; cache_path = ARGS[i]
        elseif a == "--T_frame"; i+=1; T_frame = parse(Int, ARGS[i])
        elseif a == "--frames_per_block"; i+=1; frames_per_block = parse(Int, ARGS[i])
        elseif a == "--h_len"; i+=1; h_len = parse(Int, ARGS[i])
        elseif a == "--rho"; i+=1; rho_ls = parse(Float64, ARGS[i])
        elseif a == "--Dmax"; i+=1; Dmax = parse(Int, ARGS[i])
        elseif a == "--num_data"; i+=1; num_data = parse(Int, ARGS[i])
        elseif a == "--num_repeats"; i+=1; num_repeats = parse(Int, ARGS[i])
        elseif a == "--seed"; i+=1; seed = parse(Int, ARGS[i])
        elseif a == "--no_interleaver"; use_interleaver = false
        elseif a == "--help" || a == "-h"
            println("""
Usage:
  julia --project=. scripts/make_replayswap_bases_from_raw.jl [args]

Paths:
  --rec    data/raw/logged_packets_and_ytrain.jld2
  --cache  data/ls_cache_h20_rho1e-02_bestD.jld2
  --outdir data

Geom:
  --T_frame 128
  --frames_per_block 4

Cache params:
  --h_len 20
  --rho   1e-2
  --Dmax  25

Donor params:
  --num_data 20
  --num_repeats 45

Other:
  --seed 0x2565121024
  --no_interleaver
""")
            return
        else
            error("Unknown arg: $a")
        end
        i += 1
    end
    T_block = T_frame * frames_per_block

    isfile(rec_path) || error("Missing rec file: $rec_path")
    mkpath(outdir)

    # Load recorded packets
    drec = JLD2.load(rec_path)
    haskey(drec, "all_packets_df") || error("Expected key all_packets_df in $rec_path")
    all_packets_df = DataFrame(drec["all_packets_df"])
    packet_matrix = Matrix(select(all_packets_df, Not(:frame)))
    num_frames = size(packet_matrix, 1)

    # Load or build cache
    cache = load_or_build_ls_cache(rec_path, cache_path;
                                   h_len=h_len, rho_ls=rho_ls, Dmax=Dmax,
                                   T_frame=T_frame, npc=npc, num_data=num_data, num_repeats=num_repeats)
    bestD = cache.bestD

    # Rebuild x_old per frame
    codeB, _, _, _ = initcode(64, 128, npc)
    codeB.icols === nothing && (encode(codeB, zeros(Int, 64)); nothing)
    x_old_mat = rebuild_x_old_mat(codeB; T_frame=T_frame, num_data=num_data, num_repeats=num_repeats)
    size(x_old_mat, 1) == num_frames || error("x_old_mat rows=$(size(x_old_mat,1)) != num_frames=$num_frames")

    # Interleaver (512 bits)
    rng = MersenneTwister(seed)
    π512 = use_interleaver ? randperm(rng, 512) : nothing
    π512_inv = (π512 === nothing) ? Int[] : invperm_vec(π512)

    # Codes for LDPC concat
    codeO, _, _, _ = initcode(256, 512, npc)
    codeI, _, _, _ = initcode(512, 1024, npc)
    codeO.icols === nothing && (encode(codeO, zeros(Int, 256)); nothing)
    codeI.icols === nothing && (encode(codeI, zeros(Int, 512)); nothing)

    # Block count (each block is frames_per_block frames)
    Nblk = num_frames ÷ frames_per_block
    @assert Nblk * frames_per_block == num_frames "num_frames must be multiple of frames_per_block"
    @assert T_block == 512 "This pipeline expects T_block=512 (4×128)."

    # Allocate outputs (LDPC)
    y_ldpc = Matrix{ComplexF64}(undef, Nblk, T_block)
    u256_ldpc = Matrix{Int}(undef, Nblk, 256)
    b512_ldpc = Matrix{Int}(undef, Nblk, 512)
    b512i_ldpc = Matrix{Int}(undef, Nblk, 512)
    b1024_ldpc = Matrix{Int}(undef, Nblk, 1024)
    h_blk_ldpc = Matrix{ComplexF64}(undef, Nblk, h_len)
    corr_ldpc  = Vector{Float64}(undef, Nblk)

    # Allocate outputs (RSC)
    y_rsc = Matrix{ComplexF64}(undef, Nblk, T_block)
    u256_rsc = Matrix{Int}(undef, Nblk, 256)
    b512_rsc = Matrix{Int}(undef, Nblk, 512)
    b512i_rsc = Matrix{Int}(undef, Nblk, 512)
    b1024_rsc = Matrix{Int}(undef, Nblk, 1024)
    h_blk_rsc = Matrix{ComplexF64}(undef, Nblk, h_len)
    corr_rsc  = Vector{Float64}(undef, Nblk)

    frame_ids = Matrix{Int}(undef, Nblk, frames_per_block)

    println("==============================================================")
    @printf("DFEC replay-swap base builder | frames=%d blocks=%d T_frame=%d T_block=%d\n",
            num_frames, Nblk, T_frame, T_block)
    @printf("cache: h_len=%d rho=%.2e Dmax=%d | interleaver=%s\n",
            h_len, rho_ls, Dmax, use_interleaver ? "ON" : "OFF")
    println("==============================================================")

    # Main block loop
    for b in 1:Nblk
        f0 = frames_per_block*(b-1) + 1
        fs = f0:(f0 + frames_per_block - 1)
        frame_ids[b, :] .= collect(fs)

        # build donor block y_blk, x_old_blk (apply bestD per-frame to y)
        y_parts = Vector{ComplexF64}[]
        x_parts = Vector{ComplexF64}[]
        for f in fs
            y_f = extract_symbol_rate(packet_matrix[f, :], T_frame)
            y_f = shift_left(y_f, bestD[f])
            push!(y_parts, y_f)
            push!(x_parts, ComplexF64.(vec(x_old_mat[f, :])))
        end
        y_blk = vcat(y_parts...)
        x_old_blk = vcat(x_parts...)
        @assert length(y_blk) == T_block
        @assert length(x_old_blk) == T_block

        # fit block channel h (shared for both LDPC/RSC targets)
        h_blk = ridge_ls_h(x_old_blk, y_blk, h_len, rho_ls)
        yhat_old = conv_prefix(h_blk, x_old_blk, T_block)
        corr = corr_score(y_blk, yhat_old)

        # ---- LDPC target ----
        x_new_ldpc, u256, b512, b512_i, b1024 = build_ldpc_concat_qpsk(rng, codeO, codeI, π512)
        y_new_ldpc = replay_swap_prefix(y_blk, x_old_blk, x_new_ldpc, h_blk)

        y_ldpc[b, :] .= y_new_ldpc
        u256_ldpc[b, :] .= u256
        b512_ldpc[b, :] .= b512
        b512i_ldpc[b, :] .= b512_i
        b1024_ldpc[b, :] .= b1024
        h_blk_ldpc[b, :] .= h_blk
        corr_ldpc[b] = corr

        # ---- RSC target ----
        x_new_rsc, u256r, b512r, b512ir, b1024r = build_rsc_concat_qpsk(rng, π512)
        y_new_rsc = replay_swap_prefix(y_blk, x_old_blk, x_new_rsc, h_blk)

        y_rsc[b, :] .= y_new_rsc
        u256_rsc[b, :] .= u256r
        b512_rsc[b, :] .= b512r
        b512i_rsc[b, :] .= b512ir
        b1024_rsc[b, :] .= b1024r
        h_blk_rsc[b, :] .= h_blk
        corr_rsc[b] = corr

        if b == 1 || b % 25 == 0 || b == Nblk
            @printf("blk %4d/%d | corr_donor=%.4f\n", b, Nblk, corr)
        end
    end

    # Save LDPC dataset
    out_ldpc = joinpath(outdir, "replayswap_qpsk_concat_256_512_1024_from_realdata_donorLS_h20_rho1e-2.jld2")
    meta_ldpc = (
        note = "DFEC replay-swap from raw donor capture -> LDPC concat QPSK target",
        conv_geometry = "prefix: y_clean = conv(h, x)[1:T_block]",
        T_frame = T_frame,
        frames_per_block = frames_per_block,
        T_block = T_block,
        blocks = Nblk,
        h_len = h_len,
        rho_blk = rho_ls,
        cache_path = cache_path,
        interleaver = (
            enabled = use_interleaver,
            π512 = (π512 === nothing) ? Int[] : π512,
            π512_inv = π512_inv,
        ),
    )
    @save out_ldpc y_qpsk_swapped=y_ldpc u256_mat=u256_ldpc b512_mat=b512_ldpc b512_i_mat=b512i_ldpc b1024_mat=b1024_ldpc frame_ids h_blk_mat=h_blk_ldpc corr_donor=corr_ldpc meta_out=meta_ldpc

    # Save RSC dataset
    out_rsc = joinpath(outdir, "replayswap_qpsk_RSCconcat_256_512_1024_from_realdata_donorLS_h20_rho1e-2.jld2")
    meta_rsc = (
        note = "DFEC replay-swap from raw donor capture -> RSC concat QPSK target",
        conv_geometry = "prefix: y_clean = conv(h, x)[1:T_block]",
        T_frame = T_frame,
        frames_per_block = frames_per_block,
        T_block = T_block,
        blocks = Nblk,
        h_len = h_len,
        rho_blk = rho_ls,
        cache_path = cache_path,
        interleaver = (
            enabled = use_interleaver,
            π512 = (π512 === nothing) ? Int[] : π512,
            π512_inv = π512_inv,
        ),
        rsc = (K=RSC_K, FB=RSC_FB, P=RSC_P),
    )
    @save out_rsc y_qpsk_swapped=y_rsc u256_mat=u256_rsc b512_mat=b512_rsc b512_i_mat=b512i_rsc b1024_mat=b1024_rsc frame_ids h_blk_mat=h_blk_rsc corr_donor=corr_rsc meta_out=meta_rsc

    println("\nSaved:")
    println("  LDPC → $out_ldpc")
    println("  RSC  → $out_rsc")
end

main()
