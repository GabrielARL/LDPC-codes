#!/usr/bin/env julia
# scripts/raw_dfec_oraclepilots_psweep_csv.jl
#
# RAW donor LDPC(64->128) BPSK
# Sweep ORACLE pilot ratio p = 0.0:0.1:0.5
# Evaluate only N frames per p (default N=20)
#
# Writes a tidy summary CSV compatible with psr_bpsk_1_2_psweep-style readers:
# columns: pilot_frac, method, ber, psr_pkt, psr64, nframes, lam_pil, agree_pilots
#
# Methods logged:
#   - EQ+SPA  (genie ridge-LS h using x_true; LMMSE EQ; ORACLE PILOT CLAMP into x_eq; SPA best-sign)
#   - DFEC    (decode_sparse_joint with oracle pilots + λ_pil pilot loss + pilot-phase align)
#
# Requires:
#   - lib/LDPCJDPMemoized.jl patched to support λ_pil in decode_sparse_joint
#   - data/raw/logged_packets_and_ytrain.jld2
#   - data/ls_cache_h20_rho1e-02_bestD.jld2

using Random, Printf, Statistics, LinearAlgebra
using JLD2, DataFrames, CSV
using SignalAnalysis
using SparseArrays
using Optim

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))

# IMPORTANT: avoid name collisions with other modules (e.g., LinkSim.Code)
include(joinpath(ROOT, "lib", "LDPCJDPMemoized.jl"))
import .LDPCJDPMemoized as LDM

# ----------------------------
# Small utils
# ----------------------------
@inline psr_pkt(a::AbstractVector{Int}, b::AbstractVector{Int}) = all(a .== b) ? 1.0 : 0.0
@inline ber_bits(a::AbstractVector{Int}, b::AbstractVector{Int}) = mean(a .!= b)

function psr_segments(bhat::Vector{Int}, btrue::Vector{Int}; seg::Int=64)
    n = length(bhat)
    @assert length(btrue) == n
    nseg = n ÷ seg
    nseg == 0 && return psr_pkt(bhat, btrue)
    ok = 0
    @inbounds for s in 1:nseg
        a = (s-1)*seg + 1
        b = s*seg
        ok += all(@view(bhat[a:b]) .== @view(btrue[a:b])) ? 1 : 0
    end
    return ok / nseg
end

parse_int(s::String) = parse(Int, strip(s))

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
            error("Bad --ps. Use a:b or a:step:b or comma list.")
        end
    else
        return Float64.(parse.(Float64, split(t, ",")))
    end
end

# evenly spaced pilot bit positions in 1..n
function choose_pilots_bits(n::Int; frac::Float64)
    frac <= 0 && return Int[]
    Np = max(1, round(Int, frac*n))
    posf = collect(range(1, stop=n, length=Np))
    pos = unique!(clamp.(round.(Int, posf), 1, n))
    sort!(pos)
    return pos
end

# RAW extraction matches your working scripts: row has frame index col, then n complex samples stored as numbers
function extract_symbol_rate(row::AbstractVector{T}, T_frame::Int) where {T<:Number}
    @assert length(row) >= T_frame + 1
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
        for ell in 1:min(Lh, t)
            acc += h[ell] * x[t-ell+1]
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

function lmmse_deconv_prefix(y::Vector{ComplexF64}, h::Vector{ComplexF64}, sigma2::Float64)
    T = length(y)
    Lh = length(h)
    Hc = zeros(ComplexF64, T, T)
    @inbounds for t in 1:T
        for k in 1:T
            ell = t - k + 1
            if 1 <= ell <= Lh
                Hc[t, k] = h[ell]
            end
        end
    end
    A = Hc' * Hc
    @inbounds for i in 1:T
        A[i,i] += max(sigma2, 1e-6)
    end
    rhs = Hc' * y
    return A \ rhs
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

function syndrome_weight(H::SparseMatrixCSC{Bool, Int}, bits::Vector{Int})
    m, _n = size(H)
    s = zeros(Int, m)
    I, J, _ = findnz(H)
    @inbounds for k in eachindex(I)
        s[I[k]] ⊻= (bits[J[k]] & 1)
    end
    return count(!=(0), s)
end

function spa_from_soft_bestsign(codeB,
                                colsB::Vector{Vector{Int}},
                                parityB::Vector{Vector{Int}},
                                x_soft::Vector{Float64},
                                sigma2::Float64;
                                max_iter::Int=50)

    H = LDM.get_H_sparse(codeB)

    x1, it1 = LDM.sum_product_decode(H,  x_soft, max(sigma2, 1e-6), parityB, colsB; max_iter=max_iter)
    x2, it2 = LDM.sum_product_decode(H, -x_soft, max(sigma2, 1e-6), parityB, colsB; max_iter=max_iter)

    b1 = Int.(x1); b2 = Int.(x2)
    sw1 = syndrome_weight(H, b1)
    sw2 = syndrome_weight(H, b2)

    if sw2 < sw1
        return b2, it2, -1
    elseif sw1 < sw2
        return b1, it1, +1
    else
        c1 = sum(((2 .* b1 .- 1) .* x_soft))
        c2 = sum(((2 .* b2 .- 1) .* x_soft))
        return (c2 > c1) ? (b2, it2, -1) : (b1, it1, +1)
    end
end

function argminphase_deg(xhat::AbstractVector{ComplexF64}, xref::AbstractVector{ComplexF64})
    best_deg = 0.0
    best_cost = typemax(Int)
    for deg in 0.0:0.1:360.0
        rot = exp(-im * deg2rad(deg))
        xh = xhat .* rot
        cost = sum(sign.(real.(xh)) .!= sign.(real.(xref)))
        if cost < best_cost
            best_cost = cost
            best_deg = deg
        end
    end
    return best_deg
end

function resolve_flip_by_pilots(bits::Vector{Int}, pilot_pos::Vector{Int}, pilot_bpsk::Vector{Float64})
    isempty(pilot_pos) && return bits
    vote = 0.0
    @inbounds for (k, j) in enumerate(pilot_pos)
        vote += (2*bits[j]-1) * pilot_bpsk[k]
    end
    return (vote < 0) ? (1 .- bits) : bits
end

function topk_positions(h::AbstractVector, k::Int)
    k_use = min(k, length(h))
    idx = partialsortperm(abs.(h), 1:k_use; rev=true)
    sort!(idx)
    return idx
end

# ----------------------------
# Main
# ----------------------------
function main()
    rec_path   = joinpath(DATA_DIR, "raw", "logged_packets_and_ytrain.jld2")
    cache_path = joinpath(DATA_DIR, "ls_cache_h20_rho1e-02_bestD.jld2")
    out_csv    = joinpath(DATA_DIR, "runs", "raw_dfec_oraclepilots_psweep.csv")

    ps = collect(0.0:0.1:0.5)
    start_frame = 1
    n_per_p = 20

    # Genie channel LS knobs
    h_len = 20
    rho_ls = 1e-2

    # DFEC knobs
    lam = 2.0
    lam_pil = 20.0   # pilots influence optimization
    gam = 1e-3
    eta = 1.0
    k_sparse = 4
    max_iter_opt = 20

    # CLI
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--ps"; i+=1; ps = parse_psweep(ARGS[i])
        elseif a == "--start"; i+=1; start_frame = parse_int(ARGS[i])
        elseif a == "--nperp"; i+=1; n_per_p = parse_int(ARGS[i])
        elseif a == "--lam_pil"; i+=1; lam_pil = parse(Float64, ARGS[i])
        elseif a == "--out_csv"; i+=1; out_csv = ARGS[i]
        elseif a == "--help" || a == "-h"
            println("""
Usage:
  julia --project=. scripts/raw_dfec_oraclepilots_psweep_csv.jl [args]

Args:
  --ps "0.0:0.1:0.5"     pilot ratios (default)
  --start <int>          starting frame (default 1)
  --nperp <int>          frames per p (default 20)
  --lam_pil <float>      pilot loss weight in DFEC optimizer (default 20)
  --out_csv <path>       output CSV
""")
            return
        else
            error("Unknown arg: $a")
        end
        i += 1
    end

    isfile(rec_path)   || error("Missing RAW file: $rec_path")
    isfile(cache_path) || error("Missing cache file: $cache_path")

    d  = JLD2.load(rec_path)
    all_packets_df = DataFrame(d["all_packets_df"])
    packet_matrix = Matrix(select(all_packets_df, Not(:frame)))
    num_frames = size(packet_matrix, 1)

    dc = JLD2.load(cache_path)
    bestD = Vector{Int}(dc["bestD"])
    length(bestD) == num_frames || error("bestD length mismatch")

    # code + helpers (MUST come from LDM to match decode_sparse_joint dispatch)
    codeB, colsB, _idrowsB, _ = LDM.initcode(64, 128, 4; pilot_row_fraction=0.10)
    @assert codeB isa LDM.Code "codeB is $(typeof(codeB)); name collision (did not use LDM.initcode)."

    HB = LDM.get_H_sparse(codeB)
    parityB = build_parity_indices(HB)
    n = codeB.n
    @assert n == 128

    # cw_true from SignalAnalysis.mseq(11) (matches your debug script)
    m11 = mseq(11)
    function cw_true_for_frame(frame::Int)
        idx = ((frame - 1) % 20) + 1
        bseq   = m11[idx : (codeB.k + idx - 1)]
        d_test = Int.((bseq .+ 1) ./ 2)
        cw     = LDM.encode(codeB, d_test)
        return Int.(cw)
    end

    out = DataFrame(
        pilot_frac=Float64[], method=String[],
        ber=Float64[], psr_pkt=Float64[], psr64=Float64[],
        nframes=Int[], lam_pil=Float64[], agree_pilots=Int[]
    )

    frames = collect(clamp(start_frame, 1, num_frames) : min(num_frames, start_frame + n_per_p - 1))

    println("==============================================================")
    @printf("RAW DFEC oracle-pilot PSWEEP | ps=%s | frames per p=%d | start=%d | lam_pil=%.1f\n",
            string(ps), length(frames), start_frame, lam_pil)
    println("==============================================================")

    for p in ps
        pilot_pos = choose_pilots_bits(n; frac=p)

        ber_eqL = Float64[]; psr_eqL = Float64[]; psr64_eqL = Float64[]
        ber_dfL = Float64[]; psr_dfL = Float64[]; psr64_dfL = Float64[]

        for f in frames
            y = extract_symbol_rate(packet_matrix[f, :], n)
            y = shift_left(y, bestD[f])

            cw_true = cw_true_for_frame(f)                       # 0/1
            x_true  = ComplexF64.((2 .* cw_true .- 1) .+ 0im)     # ±1

            # genie ridge-LS channel + sigma2
            h = ridge_ls_h(x_true, y, h_len, rho_ls)
            yhat = conv_prefix(h, x_true, n)
            sigma2 = Float64(mean(abs2, y .- yhat) + 1e-9)

            # EQ (LMMSE deconv)
            x_lmmse = lmmse_deconv_prefix(y, h, sigma2)
            x_eq = real.(x_lmmse)

            # ORACLE PILOT CLAMP so EQ+SPA depends on p
            if !isempty(pilot_pos)
                @inbounds for j in pilot_pos
                    x_eq[j] = (cw_true[j] == 1) ? 1.0 : -1.0
                end
            end

            # EQ+SPA baseline (best sign)
            cw_hat_eq, _it, _sgn = spa_from_soft_bestsign(codeB, colsB, parityB, x_eq, sigma2; max_iter=50)

            push!(ber_eqL, ber_bits(cw_hat_eq, cw_true))
            push!(psr_eqL, psr_pkt(cw_hat_eq, cw_true))
            push!(psr64_eqL, psr_segments(cw_hat_eq, cw_true; seg=64))

            # DFEC with oracle pilots
            pilot_bpsk = isempty(pilot_pos) ? Float64[] : Float64[(cw_true[j] == 1 ? 1.0 : -1.0) for j in pilot_pos]
            pilot_bpsk_c = ComplexF64.(pilot_bpsk .+ 0im)

            # sparse h support from genie h
            h_pos  = topk_positions(h, k_sparse)
            h_init = h[h_pos]

            _xhat_bits_bv, _hhat, optres = LDM.decode_sparse_joint(
                y, codeB, parityB, pilot_pos, pilot_bpsk_c, h_pos;
                λ=lam, λ_pil=lam_pil, γ=gam, η=eta, h_init=h_init,
                max_iter=max_iter_opt, verbose=false
            )

            # notebook-style: use z_opt soft -> align using pilots -> hard decision
            theta = Optim.minimizer(optres)
            z_opt = theta[1:n]
            x_soft = tanh.(z_opt)
            x_soft_c = ComplexF64.(x_soft .+ 0im)

            if !isempty(pilot_pos)
                ph = argminphase_deg(x_soft_c[pilot_pos], x_true[pilot_pos])
                x_soft_c .= x_soft_c .* exp(-im * deg2rad(ph))
            end

            cw_hat_df = Int.(real.(x_soft_c) .>= 0)
            cw_hat_df = resolve_flip_by_pilots(cw_hat_df, pilot_pos, pilot_bpsk)

            push!(ber_dfL, ber_bits(cw_hat_df, cw_true))
            push!(psr_dfL, psr_pkt(cw_hat_df, cw_true))
            push!(psr64_dfL, psr_segments(cw_hat_df, cw_true; seg=64))
        end

        push!(out, (p, "EQ+SPA", mean(ber_eqL), mean(psr_eqL), mean(psr64_eqL), length(frames), lam_pil, 0))
        push!(out, (p, "DFEC",   mean(ber_dfL), mean(psr_dfL), mean(psr64_dfL), length(frames), lam_pil, 0))

        @printf("p=%.2f | EQ+SPA PSR64=%.3f BER=%.4f | DFEC PSR64=%.3f BER=%.4f\n",
                p, mean(psr64_eqL), mean(ber_eqL), mean(psr64_dfL), mean(ber_dfL))
    end

    mkpath(dirname(out_csv))
    CSV.write(out_csv, out)
    println("Saved → $out_csv (rows=$(nrow(out)))")
end

main()
