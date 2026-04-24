#!/usr/bin/env julia
# scripts/debug_raw_jsdc_psweep.jl
#
# DEBUG: RAW donor LDPC(64->128) BPSK — sweep pilot ratios p=0.0:0.1:0.5 for JSDC.
#
# This version prefers the NEW robust BPSKSpinGrad core:
#   uses LinkSim.jsdc_bpsk_manual (from vendor/LinkSim/src/Codes/BPSKSpinGrad.jl)
#
# Per frame:
#   - Load RAW recording + bestD cache
#   - Rebuild x_true + cw_true using SignalAnalysis.mseq(11) (GENIE x_true for channel LS)
#   - Estimate h by ridge LS using x_true, compute σ² from residual
#   - EQ (1-shot LMMSE deconv) -> x_eq (real)
#   - EQ+SPA decode (baseline; same for all p)
#   - JSDC decode (BPSKSpinGrad) with:
#       pilot_pos  = evenly spaced positions in 1..128 (frac=p)
#       pilot_bpsk = oracle ±1 from cw_true at those positions
#
# Usage:
#   julia --project=. scripts/debug_raw_jsdc_psweep.jl
#   julia --project=. scripts/debug_raw_jsdc_psweep.jl --start 569 --nframes 200
#   julia --project=. scripts/debug_raw_jsdc_psweep.jl --psweep "0.0:0.1:0.5" --jsdc_lampil 40
#
# Optional save:
#   --out_csv data/runs/debug_raw_jsdc_psweep.csv
#
using Random, Printf, Statistics, LinearAlgebra
using JLD2, DataFrames, CSV
using SparseArrays

using DSP
using SignalAnalysis

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
const LS = ensure_linksim_loaded!()

# ---- LDPC core + decoders from LinkSim ----
using .LS: initcode, encode, get_H_sparse, sum_product_decode, Code
# ---- NEW robust BPSK JSDC core ----
using .LS: jsdc_bpsk_manual

# ----------------------------
# Small utils
# ----------------------------
@inline ber(a::AbstractVector{Int}, b::AbstractVector{Int}) = mean(a .!= b)
@inline psr_pkt(a::AbstractVector{Int}, b::AbstractVector{Int}) = all(a .== b) ? 1.0 : 0.0

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
parse_float(s::String) = parse(Float64, strip(s))

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

# evenly spaced pilot bit positions in 1..n
function choose_pilots_bits(n::Int; frac::Float64)
    frac <= 0 && return Int[]
    Np = max(1, round(Int, frac*n))
    posf = collect(range(1, stop=n, length=Np))
    pos = unique!(clamp.(round.(Int, posf), 1, n))
    sort!(pos)
    return pos
end

@inline bpsk_from_bit(b::Int) = (b == 1 ? 1.0 : -1.0)

@inline function build_parity_indices(H::SparseMatrixCSC{Bool, Int})
    m, _n = size(H)
    pi = [Int[] for _ in 1:m]
    I, J, _ = findnz(H)
    @inbounds for (i, j) in zip(I, J)
        push!(pi[i], j)
    end
    return pi
end

@inline function syndrome_weight(H::SparseMatrixCSC{Bool, Int}, bits::Vector{Int})
    m, _n = size(H)
    s = zeros(Int, m)
    I, J, _ = findnz(H)
    @inbounds for k in eachindex(I)
        s[I[k]] ⊻= (bits[J[k]] & 1)
    end
    return count(!=(0), s)
end

# ----------------------------
# RAW donor helpers (match your existing style)
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

@inline function spa_from_soft(codeB::Code,
                               colsB::Vector{Vector{Int}},
                               parityB::Vector{Vector{Int}},
                               x_soft::Vector{Float64},
                               σ2::Float64;
                               max_iter::Int=50)
    # trimmed SPA expects y and σ², and internally does L_ch=2y/σ² with hard bit (L_post<0)->1
    # For BPSK mapping bit1->+1, bit0->-1, the right sign is y_for_spa = -x_soft (existing convention)
    H = get_H_sparse(codeB)
    y_for_spa = -x_soft
    x_hat, iters = sum_product_decode(H, y_for_spa, max(σ2, 1e-6), parityB, colsB; max_iter=max_iter)
    return Int.(x_hat), iters
end

# 1-shot LMMSE deconv via normal equations (same approach as your compare script)
function lmmse_deconv_prefix(y::Vector{ComplexF64}, h::Vector{ComplexF64}, σ2::Float64)
    T = length(y)
    Lh = length(h)
    Hc = zeros(ComplexF64, T, T)
    @inbounds for t in 1:T
        for k in 1:T
            ℓ = t - k + 1
            if 1 <= ℓ <= Lh
                Hc[t, k] = h[ℓ]
            end
        end
    end
    A = Hc' * Hc
    @inbounds for i in 1:T
        A[i,i] += max(σ2, 1e-6)
    end
    rhs = Hc' * y
    return A \ rhs
end

# ----------------------------
# Main
# ----------------------------
function main()
    raw_path   = joinpath(DATA_DIR, "raw", "logged_packets_and_ytrain.jld2")
    cache_path = joinpath(DATA_DIR, "ls_cache_h20_rho1e-02_bestD.jld2")

    start = 1
    nframes = 50

    npc = 4
    T_frame = 128
    num_data = 20
    num_repeats = 45

    h_len = 20
    rho_ls = 1e-2

    spa_max_iter = 50

    # pilot sweep
    psweep = collect(0.0:0.1:0.5)

    # --- JSDC knobs (robust defaults; tweak via CLI) ---
    jsdc_max_iter = 300
    jsdc_λpar = 1.0
    jsdc_λpil = 20.0
    jsdc_λprior = 1.0
    jsdc_ηz = 1e-3
    jsdc_ηh = 1e-2
    jsdc_ηdecay = 1e-3
    jsdc_freeze_h_after = 50
    jsdc_γz = 5e-3
    jsdc_γh = 1e-3
    jsdc_h_norm_cap = 5.0
    jsdc_eps_m_par = 1e-3
    jsdc_verbose = false

    # optional save
    out_csv = ""

    # CLI
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--raw"; i+=1; raw_path = ARGS[i]
        elseif a == "--cache"; i+=1; cache_path = ARGS[i]
        elseif a == "--start"; i+=1; start = parse_int(ARGS[i])
        elseif a == "--nframes"; i+=1; nframes = parse_int(ARGS[i])
        elseif a == "--h_len"; i+=1; h_len = parse_int(ARGS[i])
        elseif a == "--rho_ls"; i+=1; rho_ls = parse_float(ARGS[i])
        elseif a == "--spa_max_iter"; i+=1; spa_max_iter = parse_int(ARGS[i])

        elseif a == "--psweep"; i+=1; psweep = parse_psweep(ARGS[i])

        # JSDC knobs
        elseif a == "--jsdc_max_iter"; i+=1; jsdc_max_iter = parse_int(ARGS[i])
        elseif a == "--jsdc_lampar"; i+=1; jsdc_λpar = parse_float(ARGS[i])
        elseif a == "--jsdc_lampil"; i+=1; jsdc_λpil = parse_float(ARGS[i])
        elseif a == "--jsdc_lamprior"; i+=1; jsdc_λprior = parse_float(ARGS[i])
        elseif a == "--jsdc_etaz"; i+=1; jsdc_ηz = parse_float(ARGS[i])
        elseif a == "--jsdc_etah"; i+=1; jsdc_ηh = parse_float(ARGS[i])
        elseif a == "--jsdc_etadecay"; i+=1; jsdc_ηdecay = parse_float(ARGS[i])
        elseif a == "--jsdc_freeze_h_after"; i+=1; jsdc_freeze_h_after = parse_int(ARGS[i])
        elseif a == "--jsdc_gamz"; i+=1; jsdc_γz = parse_float(ARGS[i])
        elseif a == "--jsdc_gamh"; i+=1; jsdc_γh = parse_float(ARGS[i])
        elseif a == "--jsdc_hcap"; i+=1; jsdc_h_norm_cap = parse_float(ARGS[i])
        elseif a == "--jsdc_eps_m"; i+=1; jsdc_eps_m_par = parse_float(ARGS[i])
        elseif a == "--jsdc_verbose"; i+=1; jsdc_verbose = (parse_int(ARGS[i]) != 0)

        elseif a == "--out_csv"; i+=1; out_csv = ARGS[i]

        elseif a == "--help" || a == "-h"
            println("""
Usage:
  julia --project=. scripts/debug_raw_jsdc_psweep.jl [args]

I/O:
  --raw <path>           (default data/raw/logged_packets_and_ytrain.jld2)
  --cache <path>         (default data/ls_cache_h20_rho1e-02_bestD.jld2)
  --start <int>          (default 1)
  --nframes <int>        (default 50)
  --out_csv <path>       (optional; saves per-(frame,p) metrics)

Sweep:
  --psweep "0.0:0.1:0.5"  (default)

Channel LS (genie x_true):
  --h_len <int>          (default 20)
  --rho_ls <float>       (default 1e-2)

SPA:
  --spa_max_iter <int>   (default 50)

JSDC (BPSKSpinGrad):
  --jsdc_max_iter <int>          (default 300)
  --jsdc_lampar <float>          (default 1.0)
  --jsdc_lampil <float>          (default 20.0)
  --jsdc_lamprior <float>        (default 1.0)
  --jsdc_etaz <float>            (default 1e-3)
  --jsdc_etah <float>            (default 1e-2)
  --jsdc_etadecay <float>        (default 1e-3)
  --jsdc_freeze_h_after <int>    (default 50)
  --jsdc_gamz <float>            (default 5e-3)
  --jsdc_gamh <float>            (default 1e-3)
  --jsdc_hcap <float>            (default 5.0)
  --jsdc_eps_m <float>           (default 1e-3)
  --jsdc_verbose 0|1
""")
            return
        else
            error("Unknown arg: $a")
        end
        i += 1
    end

    isfile(raw_path) || error("Missing RAW file: $raw_path")
    isfile(cache_path) || error("Missing cache file: $cache_path")

    d = JLD2.load(raw_path)
    all_packets_df = DataFrame(d["all_packets_df"])
    packet_matrix = Matrix(select(all_packets_df, Not(:frame)))
    num_frames = size(packet_matrix, 1)

    dc = JLD2.load(cache_path)
    bestD = Vector{Int}(dc["bestD"])
    length(bestD) == num_frames || error("cache frames=$(length(bestD)) != rec frames=$num_frames")

    codeB, colsB, _idrowsB, _ = initcode(64, 128, npc)
    codeB.icols === nothing && (encode(codeB, zeros(Int, 64)); nothing)
    HB = get_H_sparse(codeB)
    parityB = build_parity_indices(HB)

    x_old_mat, cw_true_mat = rebuild_x_old_and_cw_mat(codeB; T_frame=T_frame, num_data=num_data, num_repeats=num_repeats)
    size(x_old_mat, 1) == num_frames || error("x_old_mat rows mismatch vs frames")

    s = clamp(start, 1, num_frames)
    e = min(num_frames, s + nframes - 1)
    useN = e - s + 1

    println("==============================================================")
    @printf("DEBUG RAW JSDC PSWEEP (BPSKSpinGrad): frames %d:%d (use %d/%d)\n", s, e, useN, num_frames)
    @printf("psweep = %s\n", string(psweep))
    @printf("JSDC: max_iter=%d λpar=%.2f λpil=%.1f λprior=%.2f ηz=%.1e ηh=%.1e ηdecay=%.1e freeze_h_after=%d eps_m=%.1e\n",
            jsdc_max_iter, jsdc_λpar, jsdc_λpil, jsdc_λprior, jsdc_ηz, jsdc_ηh, jsdc_ηdecay, jsdc_freeze_h_after, jsdc_eps_m_par)
    println("==============================================================")

    out = DataFrame(
        frame=Int[], p=Float64[], bestD=Int[], sigma2=Float64[],
        ber_eqspa=Float64[], psr64_eqspa=Float64[], sw_eqspa=Int[],
        ber_jsdc=Float64[],  psr64_jsdc=Float64[],  sw_jsdc=Int[]
    )

    for f in s:e
        y = extract_symbol_rate(packet_matrix[f, :], T_frame)
        y = shift_left(y, bestD[f])

        x_true  = vec(x_old_mat[f, :])
        cw_true = Int.(vec(cw_true_mat[f, :]))

        # GENIE channel LS using x_true
        h = ridge_ls_h(x_true, y, h_len, rho_ls)
        yhat = conv_prefix(h, x_true, T_frame)
        σ2 = Float64(mean(abs2, y .- yhat))

        # EQ -> SPA baseline
        x_lmmse = lmmse_deconv_prefix(y, h, σ2)
        x_eq = real.(x_lmmse)

        cw_hat_eq, _it_spa = spa_from_soft(codeB, colsB, parityB, x_eq, σ2; max_iter=spa_max_iter)
        ber_eq = mean(cw_hat_eq .!= cw_true)
        psr64_eq = psr_segments(cw_hat_eq, cw_true; seg=64)
        sw_eq = syndrome_weight(HB, cw_hat_eq)

        # JSDC warm-start/prior: use EQ output (more ISI-aware than h1 matched filter)
        m_init = clamp.(x_eq, -0.999, 0.999)
        z_init = atanh.(m_init)
        L_prior = 2.0 .* z_init

        h_pos = collect(1:h_len)  # full window

        for p in psweep
            pilot_pos = choose_pilots_bits(T_frame; frac=p)
            pilot_bpsk = isempty(pilot_pos) ? Float64[] : Float64[bpsk_from_bit(cw_true[j]) for j in pilot_pos]

            xhat_js, _hhat, _info = jsdc_bpsk_manual(
                y, codeB, parityB, pilot_pos, pilot_bpsk, h_pos;
                λ_par=jsdc_λpar,
                λ_pil=jsdc_λpil,
                λ_prior=jsdc_λprior,
                η_z=jsdc_ηz,
                η_h=jsdc_ηh,
                η_decay=jsdc_ηdecay,
                freeze_h_after=jsdc_freeze_h_after,
                γ_z=jsdc_γz,
                γ_h=jsdc_γh,
                σ2_data=max(σ2, 1e-6),
                max_iter=jsdc_max_iter,
                h_init=h[1:h_len],
                z_init=z_init,
                L_prior=L_prior,
                max_norm_h=jsdc_h_norm_cap,
                eps_m_par=jsdc_eps_m_par,
                verbose=jsdc_verbose
            )

            cw_hat_js = Int.(xhat_js)
            ber_js = mean(cw_hat_js .!= cw_true)
            psr64_js = psr_segments(cw_hat_js, cw_true; seg=64)
            sw_js = syndrome_weight(HB, cw_hat_js)

            push!(out, (f, p, bestD[f], σ2,
                        ber_eq, psr64_eq, sw_eq,
                        ber_js, psr64_js, sw_js))
        end

        @printf("RAW frame %4d | σ2=%.3e | EQ+SPA: BER=%.3f PSR64=%.3f sw=%d | (JSDC swept)\n",
                f, σ2, ber_eq, psr64_eq, sw_eq)
    end

    println("\n==============================================================")
    for p in psweep
        sub = out[out.p .== p, :]
        @printf("MEAN p=%.2f | EQ+SPA: PSR64=%.4f BER=%.4f sw=%.1f | JSDC: PSR64=%.4f BER=%.4f sw=%.1f\n",
                p,
                mean(sub.psr64_eqspa), mean(sub.ber_eqspa), mean(sub.sw_eqspa),
                mean(sub.psr64_jsdc),  mean(sub.ber_jsdc),  mean(sub.sw_jsdc))
    end
    println("==============================================================")

    if !isempty(out_csv)
        mkpath(dirname(out_csv))
        CSV.write(out_csv, out)
        println("Saved per-(frame,p) debug CSV → $out_csv")
    end
end

main()
