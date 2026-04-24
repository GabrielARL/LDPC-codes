#!/usr/bin/env julia
# scripts/compare_rsc_and_rawldpc_plus_debug.jl
#
# CONCATENATED SCRIPT (2-in-1):
#
# MODE=compare:
#   1) RSC replay-swap BPSK dataset:
#      - decode with either:
#           (A) TurboEQ-style loop (EQ <-> BCJR)          [--rsc_method turbo]
#           (B) Linear FIR LMMSE equalizer only (NO BCJR) [--rsc_method linear]
#      - write CSV: data/runs/compare_replayswap_bpsk_RSC64_128_TURBO.csv  (single p)
#
#   2) RAW donor LDPC BPSK dataset:
#      - decode with EQ+SPA and JSDC
#      - write CSV: data/runs/compare_raw_ldpc_EQSPA_JSDC.csv
#
# MODE=debug:
#   - RSC replay-swap sanity sweep over p list, printing summaries AND
#     (optionally) saving a psweep CSV for plotting:
#       data/runs/psweep_replayswap_bpsk_RSC64_128_<method>.csv
#
# MODE=both:
#   - runs compare then debug
#
# Key upgrade for plotting:
#   - debug mode now collects per-(p,blk) rows and writes a psweep CSV
#     if --save_psweep 1 (default).
#
# Choose RSC method:
#   julia --project=. scripts/compare_rsc_and_rawldpc_plus_debug.jl --rsc_method turbo
#   julia --project=. scripts/compare_rsc_and_rawldpc_plus_debug.jl --rsc_method linear --fir_M 15 --fir_D 7 --lin_iters 2
#
# Plot:
#   After debug runs, you get:
#     data/runs/psweep_replayswap_bpsk_RSC64_128_linear.csv   (or _turbo.csv)
#   You can write/adjust a small plot script to use that CSV.
#
using Random, Printf, Statistics, LinearAlgebra
using JLD2, DataFrames, CSV
using SparseArrays

# RAW donor helpers (mseq)
using DSP
using SignalAnalysis

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
const LS = ensure_linksim_loaded!()
using .LS: initcode, encode, get_H_sparse, sum_product_decode, jsdc_qpsk_manual, Code

# RSC TurboEQ-style decoder (BPSK RSC 64->128)  (EQ <-> BCJR)
include(joinpath(ROOT, "lib", "TurboEQ_BPSK_RSC64_128.jl"))
using .TurboEQ_BPSK_RSC64_128: decode_turboeq_rsc_bpsk

# RSC Linear EQ only (NO BCJR)
include(joinpath(ROOT, "lib", "LinearEQ_BPSK_RSC64_128.jl"))
using .LinearEQ_BPSK_RSC64_128: decode_lineareq_rsc_bpsk

# ============================================================
# Common utils
# ============================================================
@inline ber(a::AbstractVector{Int}, b::AbstractVector{Int}) = mean(a .!= b)
@inline psr_pkt(a::AbstractVector{Int}, b::AbstractVector{Int}) = all(a .== b) ? 1.0 : 0.0

function parse_floats(s::String)
    t = replace(strip(s), " " => "")
    isempty(t) && return Float64[]
    return Float64.(parse.(Float64, split(t, ",")))
end

parse_int(s::String) = parse(Int, strip(s))
@inline hard_from_llr(L::AbstractVector{<:Real}) = Int.(L .< 0)

# local pilot grid helper (keeps this script independent of module exports)
function choose_pilots_bits(n::Int; frac::Float64)
    frac <= 0 && return Int[]
    Np = max(1, round(Int, frac*n))
    posf = collect(range(1, stop=n, length=Np))
    pos = unique!(clamp.(round.(Int, posf), 1, n))
    sort!(pos)
    return pos
end

# ============================================================
# RAW donor LDPC helpers (copied from your raw scripts, minimal)
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

function build_parity_indices(H::SparseMatrixCSC{Bool, Int})
    m, _n = size(H)
    pi = [Int[] for _ in 1:m]
    I, J, _ = findnz(H)
    @inbounds for (i, j) in zip(I, J)
        push!(pi[i], j)
    end
    return pi
end

# SPA wrapper (negate x_soft to match your SPA sign convention)
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
# Part 1A: RSC replay-swap compare -> CSV (single p)
# ============================================================
function run_compare_rsc(; rsc_path::String,
                         out_rsc::String,
                         corr_thr::Float64,
                         nblk_rsc::Int,
                         seed_sel::Int,
                         start::Int,
                         p_rsc::Float64,
                         turbo_iters::Int,
                         σ2_init::Float64,
                         eq_σ2_iters::Int,
                         llr_clip::Float64,
                         rsc_method::String="turbo",   # turbo | linear
                         fir_M::Int=15,
                         fir_D::Int=7,
                         lin_iters::Int=2)

    isfile(rsc_path) || error("Missing RSC dataset: $rsc_path")

    dr = JLD2.load(rsc_path)
    ymat = dr["y_bpsk_swapped"]
    umat = dr["u64_mat"]
    bmat = dr["b128_mat"]
    hmat = dr["h_blk_mat"]
    corr = dr["corr_donor"]

    nblk_all = size(ymat, 1)
    eligible = findall(corr .>= corr_thr)
    isempty(eligible) && error("No eligible RSC blocks at corr_thr=$corr_thr")

    rng = MersenneTwister(seed_sel)
    shuffle!(rng, eligible)

    start2 = clamp(start, 1, length(eligible))
    blk_list = eligible[start2 : min(length(eligible), start2 + nblk_rsc - 1)]
    isempty(blk_list) && error("Empty blk_list after selection")

    df_rsc = DataFrame(
        blk=Int[], corr=Float64[],
        p=Float64[], turbo_iters=Int[],
        u64_ber=Float64[], u64_psr=Float64[],
        b128_post_ber=Float64[], b128_post_psr=Float64[],
        b128_ch_ber=Float64[],
        sigma2_final=Float64[]
    )

    println("==============================================================")
    @printf("RSC replay-swap COMPARE | method=%s | eligible=%d using=%d/%d | corr_thr=%.2f\n",
            rsc_method, length(eligible), length(blk_list), nblk_all, corr_thr)
    if rsc_method == "turbo"
        @printf("TurboEQ: p=%.2f turbo_iters=%d σ2_init=%.3f eq_σ2_iters=%d llr_clip=%.1f\n",
                p_rsc, turbo_iters, σ2_init, eq_σ2_iters, llr_clip)
    elseif rsc_method == "linear"
        @printf("LinearEQ: p=%.2f lin_iters=%d FIR(M=%d,D=%d) σ2_init=%.3f llr_clip=%.1f\n",
                p_rsc, lin_iters, fir_M, fir_D, σ2_init, llr_clip)
    else
        error("Unknown rsc_method=$rsc_method (use turbo|linear)")
    end
    println("==============================================================")

    pilots_pos  = choose_pilots_bits(128; frac=p_rsc)

    for (ii, b) in enumerate(blk_list)
        y = ComplexF64.(vec(ymat[b, :]))
        u_true = Int.(vec(umat[b, :]))
        b_true = Int.(vec(bmat[b, :]))
        h = ComplexF64.(vec(hmat[b, :]))

        pilots_bits = isempty(pilots_pos) ? Int[] : Int.(b_true[pilots_pos])
        # NOTE: using b_true here is only non-genie if those pilot coded-bit values are truly known to the RX.

        out = if rsc_method == "turbo"
            decode_turboeq_rsc_bpsk(
                y, h, u_true, b_true;
                p=p_rsc, turbo_iters=turbo_iters, σ2_init=σ2_init,
                eq_σ2_iters=eq_σ2_iters, llr_clip=llr_clip
            )
        else
            decode_lineareq_rsc_bpsk(
                y, h;
                pilots_pos=pilots_pos,
                pilots_bits=pilots_bits,
                M=fir_M,
                D=fir_D,
                iters=lin_iters,
                σ2_init=σ2_init,
                llr_clip=llr_clip,
                u64_true=u_true,      # metrics only
                b128_true=b_true      # metrics only
            )
        end

        u_hat = out.u64_hat
        b_hat_post = hard_from_llr(out.llr128_post)
        b_hat_ch   = hard_from_llr(out.llr128_ch)

        push!(df_rsc, (
            blk=b,
            corr=Float64(corr[b]),
            p=Float64(p_rsc),
            turbo_iters=Int(turbo_iters),
            u64_ber=ber(u_hat, u_true),
            u64_psr=psr_pkt(u_hat, u_true),
            b128_post_ber=ber(b_hat_post, b_true),
            b128_post_psr=psr_pkt(b_hat_post, b_true),
            b128_ch_ber=ber(b_hat_ch, b_true),
            sigma2_final=Float64(out.sigma2_hat)
        ))

        if ii == 1 || ii % 25 == 0 || ii == length(blk_list)
            @printf("  RSC %4d/%d | corr=%.3f | u64 PSR=%.2f BER=%.3f | b128(post) BER=%.3f | b128(ch) BER=%.3f\n",
                    ii, length(blk_list), corr[b],
                    df_rsc.u64_psr[end], df_rsc.u64_ber[end],
                    df_rsc.b128_post_ber[end], df_rsc.b128_ch_ber[end])
        end
    end

    mkpath(dirname(out_rsc))
    CSV.write(out_rsc, df_rsc)
    println("Saved → $out_rsc")
    @printf("RSC mean: u64 BER=%.4f PSR=%.4f | b128(post) BER=%.4f PSR=%.4f | b128(ch) BER=%.4f\n\n",
            mean(df_rsc.u64_ber), mean(df_rsc.u64_psr),
            mean(df_rsc.b128_post_ber), mean(df_rsc.b128_post_psr),
            mean(df_rsc.b128_ch_ber))

    return df_rsc
end

# ============================================================
# Part 1B: RAW donor compare -> CSV
# ============================================================
function run_compare_raw_ldpc(; raw_path::String,
                              cache_path::String,
                              out_raw::String,
                              nframes_use::Int,
                              npc::Int,
                              T_frame::Int,
                              num_data::Int,
                              num_repeats::Int,
                              h_len::Int,
                              rho_ls::Float64,
                              jsdc_max_iter::Int,
                              jsdc_ηz::Float64,
                              jsdc_λpar::Float64,
                              jsdc_γz::Float64,
                              jsdc_γh::Float64,
                              jsdc_λprior::Float64)

    isfile(raw_path)   || error("Missing raw file: $raw_path")
    isfile(cache_path) || error("Missing cache file: $cache_path")

    d = JLD2.load(raw_path)
    haskey(d, "all_packets_df") || error("Expected key all_packets_df in $raw_path")
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
    @printf("RAW donor LDPC COMPARE | frames=%d use=%d | LDPC(64->128) BPSK\n", num_frames, useN)
    println("==============================================================")

    for f in 1:useN
        y = extract_symbol_rate(packet_matrix[f, :], T_frame)
        y = shift_left(y, bestD[f])

        x_true  = vec(x_old_mat[f, :])
        cw_true = Int.(vec(cw_true_mat[f, :]))

        # genie h (as in your original script)
        h = ridge_ls_h(x_true, y, h_len, rho_ls)

        # σ² from residual
        yhat = conv_prefix(h, x_true, T_frame)
        σ2 = Float64(mean(abs2, y .- yhat))

        # EQ+SPA (simple: 1-step LMMSE deconv via normal equations)
        Hc = zeros(ComplexF64, T_frame, T_frame)
        Lh = length(h)
        @inbounds for t in 1:T_frame
            for k in 1:T_frame
                ℓ = t - k + 1
                if 1 <= ℓ <= Lh
                    Hc[t, k] = h[ℓ]
                end
            end
        end
        A = Hc' * Hc
        @inbounds for ii in 1:T_frame
            A[ii,ii] += max(σ2, 1e-6)
        end
        rhs = Hc' * y
        x_lmmse = A \ rhs
        x_eq = real.(x_lmmse)
        cw_hat_eq, it_spa = spa_from_soft(codeB, colsB, parityB, x_eq, σ2; max_iter=50)
        ber_eqspa = mean(cw_hat_eq .!= cw_true)

        # JSDC (BPSK)
        h_pos = collect(1:h_len)
        pilot_pos = Int[]
        pilot_bpsk = Float64[]

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

    mkpath(dirname(out_raw))
    CSV.write(out_raw, df_raw)
    println("Saved → $out_raw")
    @printf("RAW mean: cw128 BER EQ+SPA=%.4f | JSDC=%.4f\n\n",
            mean(df_raw.ber_eqspa), mean(df_raw.ber_jsdc))

    return df_raw
end

# ============================================================
# Part 2: Debug/sanity sweep for RSC
#   - Prints summaries
#   - Also optionally saves a PSWEEP CSV (per-(p,blk) rows) for plotting
# ============================================================
function run_debug_rsc(; rsc_path::String,
                       corr_thr::Float64,
                       use_nblk::Int,
                       seed_sel::Int,
                       start::Int,
                       ps::Vector{Float64},
                       iters::Int,
                       σ2_init::Float64,
                       eq_σ2_iters::Int,
                       llr_clip::Float64,
                       rsc_method::String="turbo",
                       fir_M::Int=15,
                       fir_D::Int=7,
                       lin_iters::Int=2,
                       save_psweep::Bool=true,
                       out_psweep::String="")

    isfile(rsc_path) || error("Missing dataset: $rsc_path")

    d = JLD2.load(rsc_path)
    ymat = d["y_bpsk_swapped"]
    umat = d["u64_mat"]
    bmat = d["b128_mat"]
    hmat = d["h_blk_mat"]
    corr = d["corr_donor"]

    nblk_all = size(ymat, 1)

    eligible = findall(corr .>= corr_thr)
    isempty(eligible) && error("No eligible blocks at corr_thr=$corr_thr")
    rng = MersenneTwister(seed_sel)
    shuffle!(rng, eligible)

    start2 = clamp(start, 1, length(eligible))
    blk_list = eligible[start2 : min(length(eligible), start2 + use_nblk - 1)]
    isempty(blk_list) && error("Empty blk_list (start=$start)")

    println("==============================================================")
    @printf("BPSK RSC(64->128) sanity | method=%s | corr_thr=%.2f | using=%d/%d blocks\n",
            rsc_method, corr_thr, length(blk_list), nblk_all)
    if rsc_method == "turbo"
        @printf("ps = %s | turbo_iters=%d | σ2_init=%.3f | eq_σ2_iters=%d | llr_clip=%.1f\n",
                string(ps), iters, σ2_init, eq_σ2_iters, llr_clip)
    else
        @printf("ps = %s | lin_iters=%d | FIR(M=%d,D=%d) | σ2_init=%.3f | llr_clip=%.1f\n",
                string(ps), lin_iters, fir_M, fir_D, σ2_init, llr_clip)
    end
    println("==============================================================")

    # Collect per-(p,blk) rows for plotting
    df_psweep = DataFrame(
        p=Float64[],
        blk=Int[],
        corr=Float64[],
        method=String[],
        u64_psr=Float64[],
        u64_ber=Float64[],
        b128_post_psr=Float64[],
        b128_post_ber=Float64[],
        b128_ch_ber=Float64[],
        sigma2_final=Float64[]
    )

    for p in ps
        psr_u = Float64[]
        ber_u = Float64[]
        psr_b_post = Float64[]
        ber_b_post = Float64[]
        ber_b_ch = Float64[]
        sig2s = Float64[]

        pilots_pos  = choose_pilots_bits(128; frac=p)

        for (ii, b) in enumerate(blk_list)
            y = ComplexF64.(vec(ymat[b, :]))
            u_true = Int.(vec(umat[b, :]))
            b_true = Int.(vec(bmat[b, :]))
            h = ComplexF64.(vec(hmat[b, :]))

            pilots_bits = isempty(pilots_pos) ? Int[] : Int.(b_true[pilots_pos])

            out = if rsc_method == "turbo"
                decode_turboeq_rsc_bpsk(
                    y, h, u_true, b_true;
                    p=p, turbo_iters=iters, σ2_init=σ2_init,
                    eq_σ2_iters=eq_σ2_iters, llr_clip=llr_clip
                )
            else
                decode_lineareq_rsc_bpsk(
                    y, h;
                    pilots_pos=pilots_pos,
                    pilots_bits=pilots_bits,
                    M=fir_M,
                    D=fir_D,
                    iters=lin_iters,
                    σ2_init=σ2_init,
                    llr_clip=llr_clip,
                    u64_true=u_true,
                    b128_true=b_true
                )
            end

            u_hat = out.u64_hat
            b_hat_post = hard_from_llr(out.llr128_post)
            b_hat_ch   = hard_from_llr(out.llr128_ch)

            pu = psr_pkt(u_hat, u_true)
            bu = ber(u_hat, u_true)
            pb = psr_pkt(b_hat_post, b_true)
            bb = ber(b_hat_post, b_true)
            bc = ber(b_hat_ch, b_true)
            s2 = Float64(out.sigma2_hat)

            push!(psr_u, pu); push!(ber_u, bu)
            push!(psr_b_post, pb); push!(ber_b_post, bb)
            push!(ber_b_ch, bc); push!(sig2s, s2)

            if save_psweep
                push!(df_psweep, (
                    p=Float64(p),
                    blk=Int(b),
                    corr=Float64(corr[b]),
                    method=rsc_method,
                    u64_psr=Float64(pu),
                    u64_ber=Float64(bu),
                    b128_post_psr=Float64(pb),
                    b128_post_ber=Float64(bb),
                    b128_ch_ber=Float64(bc),
                    sigma2_final=Float64(s2)
                ))
            end

            if ii == 1 || ii % 50 == 0 || ii == length(blk_list)
                @printf("  p=%.2f blk %d/%d | u64 PSR=%.0f BER=%.3f | b128(post) BER=%.3f | b128(ch) BER=%.3f\n",
                        p, ii, length(blk_list),
                        psr_u[end], ber_u[end], ber_b_post[end], ber_b_ch[end])
            end
        end

        @printf("\nSUMMARY p=%.2f | u64 PSR=%.3f BER=%.3f | b128(post) PSR=%.3f BER=%.3f | b128(ch) BER=%.3f | σ2(p50)=%.4f\n\n",
                p,
                mean(psr_u), mean(ber_u),
                mean(psr_b_post), mean(ber_b_post),
                mean(ber_b_ch),
                median(sig2s))
    end

    if save_psweep
        isempty(out_psweep) && error("save_psweep=true but out_psweep is empty")
        mkpath(dirname(out_psweep))
        CSV.write(out_psweep, df_psweep)
        println("Saved RSC psweep → $out_psweep")
    end

    println("Done.")
    return df_psweep
end

# ============================================================
# Main dispatcher
# ============================================================
function main()
    # defaults
    mode = "both"  # compare|debug|both

    # RSC
    rsc_path = joinpath(DATA_DIR, "replayswap_bpsk_RSC_64_128_from_realdata_donorLS_h20_rho1e-2.jld2")
    out_rsc  = joinpath(DATA_DIR, "runs", "compare_replayswap_bpsk_RSC64_128_TURBO.csv")
    corr_thr = 0.10
    nblk_rsc = 200
    seed_sel = 12648430
    start    = 1
    p_rsc_compare = 0.0
    ps_debug = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]

    # RSC method knobs
    rsc_method = "linear"    # turbo | linear
    turbo_iters = 2          # used for turbo method
    σ2_init = 1.30
    eq_σ2_iters = 1
    llr_clip = 99.0

    # linear method knobs (FIR LMMSE)
    fir_M = 15
    fir_D = 7
    lin_iters = 2

    # Save psweep CSV for plotting (debug/both)
    save_psweep = true
    out_rsc_psweep = ""   # filled after CLI parse if empty

    # RAW
    raw_path   = joinpath(DATA_DIR, "raw", "logged_packets_and_ytrain.jld2")
    cache_path = joinpath(DATA_DIR, "ls_cache_h20_rho1e-02_bestD.jld2")
    out_raw    = joinpath(DATA_DIR, "runs", "compare_raw_ldpc_EQSPA_JSDC.csv")
    nframes_use = 200

    npc = 4
    T_frame = 128
    num_data = 20
    num_repeats = 45
    h_len = 20
    rho_ls = 1e-2

    jsdc_max_iter = 200
    jsdc_ηz = 1e-3
    jsdc_λpar = 0.1
    jsdc_γz = 5e-3
    jsdc_γh = 1e-3
    jsdc_λprior = 0.5

    # CLI parse
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a=="--mode"; i+=1; mode = ARGS[i]

        # RSC compare/debug common
        elseif a=="--rsc"; i+=1; rsc_path = ARGS[i]
        elseif a=="--out_rsc"; i+=1; out_rsc = ARGS[i]
        elseif a=="--corr"; i+=1; corr_thr = parse(Float64, ARGS[i])
        elseif a=="--nblk_rsc"; i+=1; nblk_rsc = parse_int(ARGS[i])
        elseif a=="--seed_sel"; i+=1; seed_sel = parse_int(ARGS[i])
        elseif a=="--start"; i+=1; start = parse_int(ARGS[i])

        elseif a=="--p_rsc"; i+=1; p_rsc_compare = parse(Float64, ARGS[i])
        elseif a=="--ps"; i+=1; ps_debug = parse_floats(ARGS[i])

        # RSC method + knobs
        elseif a=="--rsc_method"; i+=1; rsc_method = ARGS[i]             # turbo|linear
        elseif a=="--turbo_iters"; i+=1; turbo_iters = parse_int(ARGS[i])
        elseif a=="--lin_iters"; i+=1; lin_iters = parse_int(ARGS[i])
        elseif a=="--fir_M"; i+=1; fir_M = parse_int(ARGS[i])
        elseif a=="--fir_D"; i+=1; fir_D = parse_int(ARGS[i])

        elseif a=="--sigma2"; i+=1; σ2_init = parse(Float64, ARGS[i])
        elseif a=="--eq_sigma2_iters"; i+=1; eq_σ2_iters = parse_int(ARGS[i])
        elseif a=="--llr_clip"; i+=1; llr_clip = parse(Float64, ARGS[i])

        # Save psweep
        elseif a=="--save_psweep"; i+=1; save_psweep = (parse_int(ARGS[i]) != 0)
        elseif a=="--out_rsc_psweep"; i+=1; out_rsc_psweep = ARGS[i]

        # RAW
        elseif a=="--raw"; i+=1; raw_path = ARGS[i]
        elseif a=="--cache"; i+=1; cache_path = ARGS[i]
        elseif a=="--out_raw"; i+=1; out_raw = ARGS[i]
        elseif a=="--nframes"; i+=1; nframes_use = parse_int(ARGS[i])

        # JSDC
        elseif a=="--jsdc_max_iter"; i+=1; jsdc_max_iter = parse_int(ARGS[i])
        elseif a=="--jsdc_etaz"; i+=1; jsdc_ηz = parse(Float64, ARGS[i])
        elseif a=="--jsdc_lampar"; i+=1; jsdc_λpar = parse(Float64, ARGS[i])
        elseif a=="--jsdc_lamprior"; i+=1; jsdc_λprior = parse(Float64, ARGS[i])

        elseif a=="--help" || a=="-h"
            println("""
Usage:
  julia --project=. scripts/compare_rsc_and_rawldpc_plus_debug.jl [args]

Modes:
  --mode compare|debug|both     (default: both)

RSC dataset:
  --rsc <path>
  --out_rsc <csv>
  --corr <float>               (default 0.10)
  --nblk_rsc <int>             (default 200)
  --seed_sel <int>
  --start <int>

RSC method:
  --rsc_method turbo|linear    (default: linear)

RSC decode knobs (turbo):
  --turbo_iters <int>          (default 4)
  --eq_sigma2_iters <int>      (default 1)

RSC decode knobs (linear FIR LMMSE):
  --lin_iters <int>            (default 2)
  --fir_M <int>                (default 15)
  --fir_D <int>                (default 7)

RSC shared knobs:
  --sigma2 <float>             (default 0.30)
  --llr_clip <float>           (default 25)

RSC compare-only:
  --p_rsc <float>              (default 0.0)

RSC debug-only:
  --ps "0,0.1,0.2"             (default 0..0.5)

RSC psweep CSV (for plotting; written in debug/both):
  --save_psweep 0|1            (default 1)
  --out_rsc_psweep <csv>       (default: data/runs/psweep_replayswap_bpsk_RSC64_128_<method>.csv)

RAW donor LDPC:
  --raw <path>
  --cache <path>
  --out_raw <csv>
  --nframes <int>              (default 200, -1 = all)

JSDC:
  --jsdc_max_iter <int>
  --jsdc_etaz <float>
  --jsdc_lampar <float>
  --jsdc_lamprior <float>
""")
            return
        else
            error("Unknown arg: $a")
        end
        i += 1
    end

    # default psweep output if not provided
    if isempty(out_rsc_psweep)
        out_rsc_psweep = joinpath(DATA_DIR, "runs", "psweep_replayswap_bpsk_RSC64_128_$(lowercase(rsc_method)).csv")
    end

    if mode == "compare" || mode == "both"
        run_compare_rsc(
            rsc_path=rsc_path, out_rsc=out_rsc,
            corr_thr=corr_thr, nblk_rsc=nblk_rsc, seed_sel=seed_sel, start=start,
            p_rsc=p_rsc_compare,
            turbo_iters=turbo_iters, σ2_init=σ2_init, eq_σ2_iters=eq_σ2_iters, llr_clip=llr_clip,
            rsc_method=rsc_method, fir_M=fir_M, fir_D=fir_D, lin_iters=lin_iters
        )

        run_compare_raw_ldpc(
            raw_path=raw_path, cache_path=cache_path, out_raw=out_raw,
            nframes_use=nframes_use,
            npc=npc, T_frame=T_frame, num_data=num_data, num_repeats=num_repeats,
            h_len=h_len, rho_ls=rho_ls,
            jsdc_max_iter=jsdc_max_iter, jsdc_ηz=jsdc_ηz, jsdc_λpar=jsdc_λpar,
            jsdc_γz=jsdc_γz, jsdc_γh=jsdc_γh, jsdc_λprior=jsdc_λprior
        )
    end

    if mode == "debug" || mode == "both"
        run_debug_rsc(
            rsc_path=rsc_path,
            corr_thr=corr_thr,
            use_nblk=min(nblk_rsc, 225),
            seed_sel=seed_sel,
            start=start,
            ps=ps_debug,
            iters=turbo_iters,          # for turbo method
            σ2_init=σ2_init,
            eq_σ2_iters=eq_σ2_iters,
            llr_clip=llr_clip,
            rsc_method=rsc_method,
            fir_M=fir_M, fir_D=fir_D, lin_iters=lin_iters,
            save_psweep=save_psweep,
            out_psweep=out_rsc_psweep
        )
    end
end

main()
