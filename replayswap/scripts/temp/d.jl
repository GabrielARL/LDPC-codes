#!/usr/bin/env julia
# scripts/compare_raw_donor_3ways_nogenie_h_signfix_fecaided_signlock_flipresolve_psr.jl
#
# Same as your latest “fecaided + signlock + flipresolve” runner, plus:
#   - --start <int> to run the rest of the packets (or any range)
#   - Packet Success Rate metrics:
#       * PSR_pkt : whole cw128 correct (all bits match)
#       * PSR64   : mean success over 64-bit segments (2 segments for cw128)
#
# Run examples:
#   julia --project=. scripts/compare_raw_donor_3ways_nogenie_h_signfix_fecaided_signlock_flipresolve_psr.jl --start 1   --nframes -1
#   julia --project=. scripts/compare_raw_donor_3ways_nogenie_h_signfix_fecaided_signlock_flipresolve_psr.jl --start 201 --nframes -1
#   julia --project=. scripts/compare_raw_donor_3ways_nogenie_h_signfix_fecaided_signlock_flipresolve_psr.jl --start 401 --nframes 200

using Random, Printf, Statistics, LinearAlgebra
using SparseArrays
using JLD2, DataFrames, CSV
using DSP
using SignalAnalysis

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
const LS = ensure_linksim_loaded!()
using .LS: initcode, encode, get_H_sparse, sum_product_decode, jsdc_qpsk_manual, Code

# ----------------------------
# Helpers
# ----------------------------

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

# LMMSE equalizer for prefix-conv y ≈ H*x, with regularization σ²
function eq_lmmse_prefix(y::Vector{ComplexF64}, h::Vector{ComplexF64}, σ2::Float64)
    T = length(y)
    Hc = convmtx_prefix(h, T)
    M = Hc' * Hc
    @inbounds for i in 1:T
        M[i,i] += max(σ2, 1e-12)
    end
    rhs = Hc' * y
    return M \ rhs
end

# Rebuild TRUE donor codeword bits (for evaluation only)
function rebuild_cw_true_mat(codeB::Code; T_frame::Int, num_data::Int, num_repeats::Int)
    cw_mat = zeros(Int, num_data, T_frame)
    for i in 1:num_data
        bseq   = mseq(11)[i : (codeB.k + i - 1)]
        d_test = Int.((bseq .+ 1) ./ 2)
        cw     = encode(codeB, d_test)
        bits   = Int.(cw)
        @inbounds for t in 1:T_frame
            cw_mat[i, t] = bits[t]
        end
    end
    return repeat(cw_mat, num_repeats, 1)
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

# Count unsatisfied parity checks (lower is better; 0 = valid codeword)
function syndrome_weight(H::SparseMatrixCSC{Bool, Int}, bits::Vector{Int})
    m, _n = size(H)
    s = zeros(Int, m)
    I, J, _ = findnz(H)
    @inbounds for k in eachindex(I)
        s[I[k]] ⊻= (bits[J[k]] & 1)
    end
    return count(!=(0), s)
end

# Global flip resolve using soft reference (bits 0/1 mapped to ±1 via (2b-1))
function resolve_global_flip_by_soft(bits::Vector{Int}, x_ref::Vector{Float64})
    vote = sum(((2 .* bits .- 1) .* x_ref))
    return (vote < 0) ? (1 .- bits) : bits
end

# SPA: try both soft signs, pick smaller syndrome; if tie, break by correlation
function spa_from_soft_bestsign(codeB::Code,
                                colsB::Vector{Vector{Int}},
                                parityB::Vector{Vector{Int}},
                                x_soft::Vector{Float64},
                                σ2::Float64;
                                max_iter::Int=50)
    H = get_H_sparse(codeB)

    x1, it1 = sum_product_decode(H,  x_soft, max(σ2, 1e-6), parityB, colsB; max_iter=max_iter)
    x2, it2 = sum_product_decode(H, -x_soft, max(σ2, 1e-6), parityB, colsB; max_iter=max_iter)

    b1 = Int.(x1); b2 = Int.(x2)
    sw1 = syndrome_weight(H, b1)
    sw2 = syndrome_weight(H, b2)

    if sw2 < sw1
        return b2, it2, -1, sw2
    elseif sw1 < sw2
        return b1, it1, +1, sw1
    else
        c1 = sum(((2 .* b1 .- 1) .* x_soft))
        c2 = sum(((2 .* b2 .- 1) .* x_soft))
        if c2 > c1
            return b2, it2, -1, sw2
        else
            return b1, it1, +1, sw1
        end
    end
end

# Pseudo-pilots: only trust positions where sign(x_eq_used) agrees with cw_hat (after flip-fix)
function pick_pseudopilots_agree(x_eq_used::Vector{Float64}, cw_hat::Vector{Int}; frac::Float64=0.10)
    n = length(x_eq_used)
    @assert length(cw_hat) == n
    bpsk_hat = @. 2*cw_hat - 1

    agree = Int[]
    sizehint!(agree, n)
    @inbounds for i in 1:n
        xi = x_eq_used[i]
        si = (xi > 0) ? 1.0 : (xi < 0 ? -1.0 : 0.0)
        if si != 0.0 && si == bpsk_hat[i]
            push!(agree, i)
        end
    end
    isempty(agree) && return Int[], Float64[]

    k = clamp(round(Int, frac*n), 1, length(agree))
    mags = abs.(x_eq_used[agree])
    sel = agree[partialsortperm(mags, 1:k; rev=true)]
    pilot_pos = sort(sel)
    pilot_bpsk = [x_eq_used[i] >= 0 ? 1.0 : -1.0 for i in pilot_pos]
    return pilot_pos, pilot_bpsk
end

function pick_pseudopilots_magonly(x_soft::Vector{Float64}; frac::Float64=0.10)
    n = length(x_soft)
    k = clamp(round(Int, frac*n), 1, n)
    mags = abs.(x_soft)
    idx = partialsortperm(mags, 1:k; rev=true)
    pilot_pos = sort!(collect(idx))
    pilot_bpsk = [x_soft[i] >= 0 ? 1.0 : -1.0 for i in pilot_pos]
    return pilot_pos, pilot_bpsk
end

# Resolve global flip using pilots targets directly.
function resolve_global_flip_by_pilots(bits::Vector{Int}, pilot_pos::Vector{Int}, pilot_bpsk::Vector{Float64})
    isempty(pilot_pos) && return bits
    vote = 0.0
    @inbounds for (k, j) in enumerate(pilot_pos)
        vote += (2*bits[j]-1) * pilot_bpsk[k]
    end
    return (vote < 0) ? (1 .- bits) : bits
end

# Resolve global bit flip by correlating with soft x_ref
function resolve_global_flip(bits::Vector{Int}, x_ref::Vector{Float64})
    vote = sum(((2 .* bits .- 1) .* x_ref))
    return (vote < 0) ? (1 .- bits) : bits
end

# Pick a short window around strongest tap (reduces JSDC degrees of freedom)
function window_h_by_max(h::Vector{ComplexF64}, Ls::Int)
    Ls <= 0 && return copy(h)
    Ls = min(Ls, length(h))
    ℓ0 = argmax(abs.(h))
    a = max(1, ℓ0 - (Ls ÷ 2))
    b = min(length(h), a + Ls - 1)
    a = max(1, b - Ls + 1)
    return h[a:b]
end

# Packet success rate: whole packet correct
@inline psr_pkt(bhat::Vector{Int}, btrue::Vector{Int}) = all(bhat .== btrue) ? 1.0 : 0.0

# PSR over fixed-size segments (e.g., 64-bit chunks)
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

# ----------------------------
# DD channel estimation (BPSK)
# ----------------------------

function estimate_h_dd_bpsk(y::Vector{ComplexF64};
                            h_len::Int=20,
                            rho::Float64=1e-2,
                            iters::Int=3)
    T = length(y)
    xhat = Vector{ComplexF64}(undef, T)
    @inbounds for t in 1:T
        xhat[t] = (real(y[t]) >= 0 ? 1.0 : -1.0) + 0im
    end

    σ2 = 0.5 * mean(abs2, y) + 1e-6
    h = zeros(ComplexF64, h_len)
    x_soft_c = ComplexF64.(xhat)

    for _ in 1:iters
        X = zeros(ComplexF64, T, h_len)
        @inbounds for t in 1:T
            for k in 1:h_len
                idx = t - k + 1
                X[t, k] = idx >= 1 ? xhat[idx] : (0.0 + 0im)
            end
        end

        A = X'X
        @inbounds for i in 1:h_len
            A[i,i] += rho
        end
        b = X'y
        h = A \ b

        yhat = conv_prefix(h, xhat, T)
        e = y .- yhat
        σ2 = mean(abs2, e) + 1e-9

        x_soft_c = eq_lmmse_prefix(y, h, σ2)
        @inbounds for t in 1:T
            xhat[t] = (real(x_soft_c[t]) >= 0 ? 1.0 : -1.0) + 0im
        end
    end

    return Vector{ComplexF64}(h), Float64(σ2), Vector{ComplexF64}(x_soft_c)
end

# Re-estimate h,σ² from decoded cw bits (0/1) => BPSK symbols (±1)
function estimate_h_from_bits(y::Vector{ComplexF64},
                              bits::Vector{Int};
                              h_len::Int=20,
                              rho::Float64=1e-2)
    T = length(y)
    @assert length(bits) == T
    x_ref = ComplexF64.((2 .* bits .- 1) .+ 0im)

    X = zeros(ComplexF64, T, h_len)
    @inbounds for t in 1:T
        for k in 1:h_len
            idx = t - k + 1
            X[t, k] = idx >= 1 ? x_ref[idx] : (0.0 + 0im)
        end
    end

    A = X'X
    @inbounds for i in 1:h_len
        A[i,i] += rho
    end
    b = X' * y
    h = A \ b

    yhat = conv_prefix(Vector{ComplexF64}(h), x_ref, T)
    e = y .- yhat
    σ2 = mean(abs2, e) + 1e-9
    return Vector{ComplexF64}(h), Float64(σ2)
end

# FEC-aided DD with SIGN-LOCK + FLIP-RESOLVE (uses oriented x_eq_used each round)
function fec_aided_dd(y::Vector{ComplexF64},
                      codeB::Code,
                      colsB::Vector{Vector{Int}},
                      parityB::Vector{Vector{Int}};
                      h_len::Int=20,
                      rho::Float64=1e-2,
                      dd_iters::Int=2,
                      outer_rounds::Int=3,
                      max_iter_spa::Int=50)

    h, σ2, _ = estimate_h_dd_bpsk(y; h_len=h_len, rho=rho, iters=dd_iters)
    h_prev = copy(h)

    for _r in 1:outer_rounds
        x_soft_c = eq_lmmse_prefix(y, h, σ2)
        x_eq_raw = real.(x_soft_c)

        cw_hat, _it, sgn_eq, _sw = spa_from_soft_bestsign(codeB, colsB, parityB, x_eq_raw, σ2; max_iter=max_iter_spa)
        x_eq_used = sgn_eq * x_eq_raw

        # orient cw to match oriented soft
        cw_hat = resolve_global_flip_by_soft(cw_hat, x_eq_used)

        h_new, σ2_new = estimate_h_from_bits(y, cw_hat; h_len=h_len, rho=rho)

        # sign-lock on h
        dpos = norm(h_new .- h_prev)
        dneg = norm((-h_new) .- h_prev)
        if dneg < dpos
            h_new .*= -1
            cw_hat .= 1 .- cw_hat
        end

        h_prev .= h_new
        h .= h_new
        σ2 = σ2_new
    end

    return h, σ2
end

# ----------------------------
# JSDC: try both signs and pick by syndrome (WITH PILOTS)
# ----------------------------

function jsdc_bestsign(y::Vector{ComplexF64},
                       codeB::Code,
                       H::SparseMatrixCSC{Bool, Int},
                       parityB::Vector{Vector{Int}},
                       h_init::Vector{ComplexF64},
                       z_init::Vector{Float64},
                       L_prior::Vector{Float64},
                       σ2::Float64,
                       pilot_pos::Vector{Int},
                       pilot_bpsk::Vector{Float64};
                       λ_par::Float64=1.0,
                       λ_pil::Float64=20.0,
                       λ_prior::Float64=1.0,
                       η_z::Float64=3e-4,
                       max_iter::Int=600,
                       γ_z::Float64=5e-3,
                       γ_h::Float64=1e-3)

    h_pos = collect(1:length(h_init))

    xA, _, _ = jsdc_qpsk_manual(
        y, codeB, parityB, pilot_pos, pilot_bpsk, h_pos;
        modulation=:bpsk,
        λ_par=λ_par,
        λ_pil=λ_pil,
        γ_z=γ_z,
        γ_h=γ_h,
        η_z=η_z,
        η_h=0.0,
        max_iter=max_iter,
        h_init=h_init,
        z_init=z_init,
        L_prior=L_prior,
        λ_prior=λ_prior,
        σ2_data=σ2,
        verbose=false
    )
    bA = Int.(xA)
    swA = syndrome_weight(H, bA)

    xB, _, _ = jsdc_qpsk_manual(
        y, codeB, parityB, pilot_pos, pilot_bpsk, h_pos;
        modulation=:bpsk,
        λ_par=λ_par,
        λ_pil=λ_pil,
        γ_z=γ_z,
        γ_h=γ_h,
        η_z=η_z,
        η_h=0.0,
        max_iter=max_iter,
        h_init=h_init,
        z_init=-z_init,
        L_prior=-L_prior,
        λ_prior=λ_prior,
        σ2_data=σ2,
        verbose=false
    )
    bB = Int.(xB)
    swB = syndrome_weight(H, bB)

    if swB < swA
        return bB, -1, swB
    else
        return bA, +1, swA
    end
end

# ----------------------------
# Main
# ----------------------------

function main()
    rec_path   = joinpath(DATA_DIR, "raw", "logged_packets_and_ytrain.jld2")
    cache_path = joinpath(DATA_DIR, "ls_cache_h20_rho1e-02_bestD.jld2")
    outcsv     = joinpath(DATA_DIR, "runs", "compare_raw_donor_3ways_nogenie_h_signfix_fecaided_signlock_flipresolve_psr.csv")

    # must match capture generation
    npc = 4
    T_frame = 128
    num_data = 20
    num_repeats = 45

    # DD / FEC-aided channel estimation params
    h_len = 20
    rho_dd = 1e-2
    dd_iters = 2
    outer_rounds = 3

    # range
    start_frame = 1
    nframes_use = -1   # -1 = all

    # pseudo-pilot knobs
    pilot_frac = 0.10
    λ_pil = 20.0
    use_agree_pilots = 1  # 1 = agree-with-SPA, 0 = mag-only

    # JSDC knobs
    jsdc_max_iter = 600
    jsdc_ηz = 3e-4
    jsdc_λpar = 1.0
    jsdc_γz = 5e-3
    jsdc_γh = 1e-3
    jsdc_λprior = 1.0
    h_jsdc_Ls = 7         # 0 disables; else strongest-tap window length

    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--rec"; i+=1; rec_path = ARGS[i]
        elseif a == "--cache"; i+=1; cache_path = ARGS[i]
        elseif a == "--outcsv"; i+=1; outcsv = ARGS[i]

        elseif a == "--start"; i+=1; start_frame = parse(Int, ARGS[i])
        elseif a == "--nframes"; i+=1; nframes_use = parse(Int, ARGS[i])

        elseif a == "--h_len"; i+=1; h_len = parse(Int, ARGS[i])
        elseif a == "--rho_dd"; i+=1; rho_dd = parse(Float64, ARGS[i])
        elseif a == "--dd_iters"; i+=1; dd_iters = parse(Int, ARGS[i])
        elseif a == "--outer_rounds"; i+=1; outer_rounds = parse(Int, ARGS[i])

        elseif a == "--pilot_frac"; i+=1; pilot_frac = parse(Float64, ARGS[i])
        elseif a == "--lam_pil"; i+=1; λ_pil = parse(Float64, ARGS[i])
        elseif a == "--agree_pilots"; i+=1; use_agree_pilots = parse(Int, ARGS[i])

        elseif a == "--h_jsdc_Ls"; i+=1; h_jsdc_Ls = parse(Int, ARGS[i])

        elseif a == "--jsdc_max_iter"; i+=1; jsdc_max_iter = parse(Int, ARGS[i])
        elseif a == "--jsdc_etaz"; i+=1; jsdc_ηz = parse(Float64, ARGS[i])
        elseif a == "--jsdc_lampar"; i+=1; jsdc_λpar = parse(Float64, ARGS[i])
        elseif a == "--jsdc_lamprior"; i+=1; jsdc_λprior = parse(Float64, ARGS[i])

        elseif a == "--help" || a == "-h"
            println("help omitted for brevity")
            return
        else
            error("Unknown arg: $a")
        end
        i += 1
    end

    isfile(rec_path)   || error("Missing rec file: $rec_path")
    isfile(cache_path) || error("Missing cache file: $cache_path (timing cache)")
    mkpath(dirname(outcsv))

    # Load recorded packets
    d = JLD2.load(rec_path)
    haskey(d, "all_packets_df") || error("Expected key all_packets_df in $rec_path")
    all_packets_df = DataFrame(d["all_packets_df"])
    packet_matrix = Matrix(select(all_packets_df, Not(:frame)))
    num_frames = size(packet_matrix, 1)

    # Load timing cache (bestD)
    dc = JLD2.load(cache_path)
    bestD = Vector{Int}(dc["bestD"])
    length(bestD) == num_frames || error("cache frames=$(length(bestD)) != rec frames=$num_frames")

    # LDPC code (64->128)
    codeB, colsB, idrowsB, _ = initcode(64, 128, npc)
    codeB.icols === nothing && (encode(codeB, zeros(Int, 64)); nothing)
    H = get_H_sparse(codeB)
    parityB = build_parity_indices(H)

    # TRUE cw bits for evaluation
    cw_true_mat = rebuild_cw_true_mat(codeB; T_frame=T_frame, num_data=num_data, num_repeats=num_repeats)
    size(cw_true_mat, 1) == num_frames || error("cw_true_mat rows mismatch")

    start_frame = clamp(start_frame, 1, num_frames)
    end_frame = (nframes_use < 0) ? num_frames : min(num_frames, start_frame + nframes_use - 1)
    useN = end_frame - start_frame + 1

    df = DataFrame(
        frame=Int[],
        bestD=Int[],
        sigma2=Float64[],

        ber_spa=Float64[],
        ber_eq=Float64[],
        ber_jsdc=Float64[],

        psr_spa_pkt=Float64[],
        psr_eq_pkt=Float64[],
        psr_jsdc_pkt=Float64[],

        psr_spa64=Float64[],
        psr_eq64=Float64[],
        psr_jsdc64=Float64[],

        spa_iters=Int[],
        eq_iters=Int[],
        spa_sgn=Int[],
        eq_sgn=Int[],
        jsdc_sgn=Int[],
        spa_synd=Int[],
        eq_synd=Int[],
        jsdc_synd=Int[],
        npil=Int[],
        lam_pil=Float64[],
        pilot_frac=Float64[],
        jsdc_flip_pil=Int[],
        jsdc_flip_eq=Int[],
        outer_rounds=Int[],
        agree_pilots=Int[],
        h_jsdc_Ls=Int[],
    )

    println("==============================================================")
    @printf("RAW DONOR 3-WAY (NO-GENIE + FEC-AIDED h,σ² + SIGNFIX + PSEUDO-PILOTS + FLIPFIX)\n")
    @printf("frames=%d use=%d | range=%d:%d | LDPC(64->128) BPSK\n", num_frames, useN, start_frame, end_frame)
    @printf("cache(bestD)=%s\n", cache_path)
    @printf("FEC-aided: h_len=%d rho=%g dd_iters=%d outer_rounds=%d\n", h_len, rho_dd, dd_iters, outer_rounds)
    @printf("Pseudo pilots: frac=%.3f  λ_pil=%g  agree_pilots=%d\n", pilot_frac, λ_pil, use_agree_pilots)
    @printf("JSDC: ηz=%g λpar=%g λprior=%g it=%d | h_jsdc_Ls=%d\n", jsdc_ηz, jsdc_λpar, jsdc_λprior, jsdc_max_iter, h_jsdc_Ls)
    println("==============================================================")

    for f in start_frame:end_frame
        y = extract_symbol_rate(packet_matrix[f, :], T_frame)
        y = shift_left(y, bestD[f])
        cw_true = Int.(vec(cw_true_mat[f, :]))

        # --- FEC-aided channel/noise estimation ---
        h_dd, σ2 = fec_aided_dd(y, codeB, colsB, parityB; h_len=h_len, rho=rho_dd, dd_iters=dd_iters, outer_rounds=outer_rounds)

        # Final EQ soft using refined h,σ²
        x_soft_c = eq_lmmse_prefix(y, h_dd, σ2)

        # (A) MF -> SPA (flip-resolve for reporting)
        h1 = h_dd[1]
        x_mf = real.(conj(h1) .* y) ./ max(abs2(h1), 1e-12)
        xhat_spa, it_spa, sgn_spa, sw_spa = spa_from_soft_bestsign(codeB, colsB, parityB, x_mf, σ2; max_iter=50)
        x_mf_used = sgn_spa * x_mf
        xhat_spa = resolve_global_flip_by_soft(xhat_spa, x_mf_used)
        ber_spa = mean(xhat_spa .!= cw_true)

        # (B) EQ -> SPA (flip-resolve for reporting + pilots)
        x_eq_raw = real.(x_soft_c)
        xhat_eq, it_eq, sgn_eq, sw_eq = spa_from_soft_bestsign(codeB, colsB, parityB, x_eq_raw, σ2; max_iter=50)
        x_eq_used = sgn_eq * x_eq_raw
        xhat_eq = resolve_global_flip_by_soft(xhat_eq, x_eq_used)
        ber_eq = mean(xhat_eq .!= cw_true)

        # PSR metrics for SPA/EQ
        psr_spa_pkt = psr_pkt(xhat_spa, cw_true)
        psr_eq_pkt  = psr_pkt(xhat_eq,  cw_true)
        psr_spa64   = psr_segments(xhat_spa, cw_true; seg=64)
        psr_eq64    = psr_segments(xhat_eq,  cw_true; seg=64)

        # (C) JSDC: pseudo-pilots + warm-start from SIGN-CORRECTED EQ soft
        pilot_pos, pilot_bpsk = if use_agree_pilots != 0
            pick_pseudopilots_agree(x_eq_used, xhat_eq; frac=pilot_frac)
        else
            pick_pseudopilots_magonly(x_eq_used; frac=pilot_frac)
        end

        m_init = clamp.(x_eq_used, -0.999, 0.999)
        z_init = atanh.(m_init)
        L_prior = 2.0 .* z_init

        h_for_jsdc = (h_jsdc_Ls > 0) ? window_h_by_max(h_dd, h_jsdc_Ls) : h_dd

        xhat_jsdc, sgn_jsdc, _sw_jsdc = jsdc_bestsign(
            y, codeB, H, parityB, h_for_jsdc, z_init, L_prior, σ2,
            pilot_pos, pilot_bpsk;
            λ_par=jsdc_λpar,
            λ_pil=λ_pil,
            λ_prior=jsdc_λprior,
            η_z=jsdc_ηz,
            max_iter=jsdc_max_iter,
            γ_z=jsdc_γz,
            γ_h=jsdc_γh
        )

        # JSDC flip-resolve (pilots then soft vote)
        x_after_pil = resolve_global_flip_by_pilots(xhat_jsdc, pilot_pos, pilot_bpsk)
        flip_pil = any(x_after_pil .!= xhat_jsdc) ? 1 : 0
        x_after_eq = resolve_global_flip(x_after_pil, x_eq_used)
        flip_eq = any(x_after_eq .!= x_after_pil) ? 1 : 0
        xhat_jsdc = x_after_eq

        sw_jsdc2 = syndrome_weight(H, xhat_jsdc)
        ber_jsdc = mean(xhat_jsdc .!= cw_true)

        psr_jsdc_pkt = psr_pkt(xhat_jsdc, cw_true)
        psr_jsdc64   = psr_segments(xhat_jsdc, cw_true; seg=64)

        push!(df, (
            frame=f,
            bestD=bestD[f],
            sigma2=σ2,

            ber_spa=ber_spa,
            ber_eq=ber_eq,
            ber_jsdc=ber_jsdc,

            psr_spa_pkt=psr_spa_pkt,
            psr_eq_pkt=psr_eq_pkt,
            psr_jsdc_pkt=psr_jsdc_pkt,

            psr_spa64=psr_spa64,
            psr_eq64=psr_eq64,
            psr_jsdc64=psr_jsdc64,

            spa_iters=it_spa,
            eq_iters=it_eq,
            spa_sgn=sgn_spa,
            eq_sgn=sgn_eq,
            jsdc_sgn=sgn_jsdc,
            spa_synd=sw_spa,
            eq_synd=sw_eq,
            jsdc_synd=sw_jsdc2,
            npil=length(pilot_pos),
            lam_pil=λ_pil,
            pilot_frac=pilot_frac,
            jsdc_flip_pil=flip_pil,
            jsdc_flip_eq=flip_eq,
            outer_rounds=outer_rounds,
            agree_pilots=use_agree_pilots,
            h_jsdc_Ls=h_jsdc_Ls,
        ))

        # progress prints
        if f == start_frame || ((f - start_frame + 1) % 25 == 0) || f == end_frame
            @printf("frame %4d/%d | BER: SPA=%.3f EQ=%.3f JSDC=%.3f | PSR(pkt): S=%.2f E=%.2f J=%.2f | PSR64: S=%.2f E=%.2f J=%.2f | σ2=%.3e | synd: S=%d E=%d J=%d | npil=%d | sgn_eq=%d | flip(pil,eq)=(%d,%d)\n",
                f, end_frame,
                ber_spa, ber_eq, ber_jsdc,
                psr_spa_pkt, psr_eq_pkt, psr_jsdc_pkt,
                psr_spa64, psr_eq64, psr_jsdc64,
                σ2, sw_spa, sw_eq, sw_jsdc2,
                length(pilot_pos), sgn_eq, flip_pil, flip_eq)
        end
    end

    CSV.write(outcsv, df)
    println("Saved → $outcsv")

    println("\n---------------- Summary ----------------")
    @printf("Mean BER (cw128): SPA=%.4f  EQ=%.4f  JSDC=%.4f\n",
            mean(df.ber_spa), mean(df.ber_eq), mean(df.ber_jsdc))
    @printf("PSR(pkt)        : SPA=%.4f  EQ=%.4f  JSDC=%.4f\n",
            mean(df.psr_spa_pkt), mean(df.psr_eq_pkt), mean(df.psr_jsdc_pkt))
    @printf("PSR64           : SPA=%.4f  EQ=%.4f  JSDC=%.4f\n",
            mean(df.psr_spa64), mean(df.psr_eq64), mean(df.psr_jsdc64))
end

main()
