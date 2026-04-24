#!/usr/bin/env julia
# scripts/psr_bpsk_1_2.jl
#
# Sweeps pilot ratio p for BOTH:
#
# RSC replay-swap BPSK RSC(64->128):
#   - LinearEQ baseline (no BCJR)
#   - (Optional) TurboEQ (EQ<->BCJR loop)
#
# RAW donor LDPC BPSK LDPC(64->128):
#   - EQ+SPA (NON-ORACLE, truly sweeps with p):
#       1) crude DD init x0 = sign(real(y))
#       2) estimate h0 by ridge LS
#       3) equalize -> x_eq0
#       4) for each p: choose pseudo-pilots = top-|x_eq0| fraction
#          do weighted ridge LS emphasizing pilot positions -> h_ref(p)
#          equalize with h_ref(p) -> LLR -> SPA
#
#   - JSDC (oracle pilots) swept over p
#   - JSDC -> SPA (finish with SPA using JSDC soft output) swept over p
#
# Outputs:
#   data/runs/psr_bpsk_1_2_rsc_detail.csv
#   data/runs/psr_bpsk_1_2_raw_detail.csv
#   data/runs/psr_bpsk_1_2_summary_tidy.csv
#
using Random, Printf, Statistics, LinearAlgebra
using JLD2, DataFrames, CSV
using SparseArrays

using DSP
using SignalAnalysis

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
const LS = ensure_linksim_loaded!()

# LinkSim core bits (trimmed LDPC + SPA)
using .LS: initcode, encode, get_H_sparse, sum_product_decode, Code

# Prefer BPSKSpinGrad if present; else fall back to QPSKSpinGrad(:bpsk)
const HAVE_JSDC_BPSK = isdefined(LS, :jsdc_bpsk_manual)
if !HAVE_JSDC_BPSK
    using .LS: jsdc_qpsk_manual
else
    @info "Using LinkSim.jsdc_bpsk_manual (BPSKSpinGrad)"
end

# RSC decoders
include(joinpath(ROOT, "lib", "TurboEQ_BPSK_RSC64_128.jl"))
using .TurboEQ_BPSK_RSC64_128: decode_turboeq_rsc_bpsk
include(joinpath(ROOT, "lib", "LinearEQ_BPSK_RSC64_128.jl"))
using .LinearEQ_BPSK_RSC64_128: decode_lineareq_rsc_bpsk

# ============================================================
# Common utils
# ============================================================
@inline hard_from_llr(L::AbstractVector{<:Real}) = Int.(L .< 0)
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
            error("Bad sweep. Use a:b or a:step:b or comma list.")
        end
    else
        return Float64.(parse.(Float64, split(t, ",")))
    end
end

function choose_pilots_bits(n::Int; frac::Float64)
    frac <= 0 && return Int[]
    Np = max(1, round(Int, frac*n))
    posf = collect(range(1, stop=n, length=Np))
    pos = unique!(clamp.(round.(Int, posf), 1, n))
    sort!(pos)
    return pos
end

@inline bpsk_from_bit(b::Int) = (b == 1 ? 1.0 : -1.0)

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

# ============================================================
# Tidy summary helper
# ============================================================
function melt_metrics(sumdf::DataFrame; domain::String, xname::String, xcol::Symbol, methodcol::Symbol, metrics::Vector{Symbol})
    out = DataFrame(domain=String[], xname=String[], xval=Float64[], method=String[], metric=String[], value=Float64[])
    for r in eachrow(sumdf)
        for m in metrics
            push!(out, (domain, xname, Float64(r[xcol]), String(r[methodcol]), String(m), Float64(r[m])))
        end
    end
    return out
end

# ============================================================
# RAW donor helpers (debug/genie packet construction, but NON-ORACLE EQ+SPA)
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

# Build Toeplitz conv matrix for x (prefix model): y[t] = Σ h[k] x[t-k+1]
function convmtx_x_prefix(x::Vector{ComplexF64}, h_len::Int)
    T = length(x)
    X = zeros(ComplexF64, T, h_len)
    @inbounds for t in 1:T
        for k in 1:h_len
            idx = t - k + 1
            X[t, k] = (idx >= 1) ? x[idx] : (0.0 + 0im)
        end
    end
    return X
end

# Weighted ridge LS channel estimate:
#   min_h Σ w[t]*| (Xh - y)[t] |^2 + rho*||h||^2
function ridge_wls_h_from_x(y::Vector{ComplexF64},
                            x_ref::Vector{ComplexF64},
                            h_len::Int,
                            rho::Float64,
                            pilot_pos::Vector{Int},
                            w_pil::Float64)
    @assert length(y) == length(x_ref)
    T = length(y)
    X = convmtx_x_prefix(x_ref, h_len)

    w = ones(Float64, T)
    if !isempty(pilot_pos) && w_pil > 0
        @inbounds for j in pilot_pos
            1 <= j <= T && (w[j] += w_pil)
        end
    end

    # compute X' W X and X' W y without forming W
    A = zeros(ComplexF64, h_len, h_len)
    b = zeros(ComplexF64, h_len)

    @inbounds for t in 1:T
        wt = w[t]
        xt = @view X[t, :]
        yt = y[t]
        # b += wt * conj(xt) * yt
        for i in 1:h_len
            b[i] += wt * conj(xt[i]) * yt
        end
        # A += wt * conj(xt)' * xt
        for i in 1:h_len
            cxi = conj(xt[i])
            for j in 1:h_len
                A[i,j] += wt * cxi * xt[j]
            end
        end
    end
    @inbounds for i in 1:h_len
        A[i,i] += rho
    end
    return Vector{ComplexF64}(A \ b)
end

# LMMSE deconv using prefix conv matrix H(h)
function eq_lmmse_deconv(y::Vector{ComplexF64}, h::Vector{ComplexF64}, σ2::Float64)
    T = length(y)
    Hc = convmtx_prefix(h, T)
    A = Hc' * Hc
    @inbounds for i in 1:T
        A[i,i] += max(σ2, 1e-12)
    end
    return A \ (Hc' * y)
end

# Non-oracle pseudo-pilots: top-|x_eq| positions, bit from sign
function pseudo_pilots_from_xeq(x_eq::Vector{Float64}; frac::Float64)
    n = length(x_eq)
    frac <= 0 && return Int[], Int[]
    k = clamp(round(Int, frac*n), 1, n)
    idx = partialsortperm(abs.(x_eq), 1:k; rev=true)
    pos = sort!(collect(idx))
    bits = [x_eq[i] >= 0 ? 1 : 0 for i in pos]
    return pos, bits
end

# Channel L01 (log P0/P1) for bit1->+1, bit0->-1: L01 ≈ -2*x/σ²
function bpsk_L01_from_xeq(x_eq::Vector{Float64}, σ2::Float64; clipL::Float64=25.0)
    c = -2.0 / max(σ2, 1e-12)
    L01 = c .* x_eq
    @inbounds for i in eachindex(L01)
        L01[i] = clamp(L01[i], -clipL, clipL)
    end
    return L01
end

# Feed LLRs directly into trimmed SPA:
# SPA uses L_ch = 2*y/σ² internally, so set y=0.5*LLR and σ²=1.0.
function spa_from_llr(codeB::Code,
                      colsB::Vector{Vector{Int}},
                      parityB::Vector{Vector{Int}},
                      L01::Vector{Float64};
                      max_iter::Int=50)
    H = get_H_sparse(codeB)
    y_for_spa = 0.5 .* L01
    x_hat, iters = sum_product_decode(H, y_for_spa, 1.0, parityB, colsB; max_iter=max_iter)
    return Int.(x_hat), iters
end

# JSDC->SPA conversion: m -> L01 = -2atanh(m)
function llr01_from_m(m::AbstractVector{<:Real}; clip_m::Float64=0.999999, clip_L::Float64=25.0)
    mm = Float64.(m)
    @inbounds for i in eachindex(mm)
        mm[i] = clamp(mm[i], -clip_m, clip_m)
    end
    L01 = -2.0 .* atanh.(mm)
    @inbounds for i in eachindex(L01)
        L01[i] = clamp(L01[i], -clip_L, clip_L)
    end
    return L01
end

# Global flip resolve (syndrome primary; pilot vote tie-break)
function resolve_flip(bits::Vector{Int}, H::SparseMatrixCSC{Bool,Int},
                      pilot_pos::Vector{Int}=Int[], pilot_bpsk::Vector{Float64}=Float64[])
    b1 = Vector{Int}(bits)
    b2 = 1 .- b1
    sw1 = syndrome_weight(H, b1)
    sw2 = syndrome_weight(H, b2)
    if sw2 < sw1
        return b2
    elseif sw1 < sw2
        return b1
    end
    if !isempty(pilot_pos)
        v = 0.0
        @inbounds for (k, j) in enumerate(pilot_pos)
            v += (2*b1[j]-1) * pilot_bpsk[k]
        end
        return (v < 0) ? b2 : b1
    end
    return b1
end

# ============================================================
# RUN RSC (sweep p)
# ============================================================
function run_rsc_detail(; rsc_path::String,
                        out_csv::String,
                        psweep_rsc::Vector{Float64},
                        rsc_methods::Vector{String},
                        corr_thr::Float64,
                        nblk_rsc::Int,
                        seed_sel::Int,
                        start::Int,
                        turbo_iters::Int,
                        σ2_init_turbo::Float64,
                        eq_σ2_iters::Int,
                        llr_clip::Float64,
                        fir_M::Int,
                        fir_D::Int,
                        lin_iters::Int,
                        σ2_init_lin::Float64)

    isfile(rsc_path) || error("Missing RSC dataset: $rsc_path")
    d = JLD2.load(rsc_path)
    ymat = d["y_bpsk_swapped"]; umat = d["u64_mat"]; bmat = d["b128_mat"]; hmat = d["h_blk_mat"]; corr = d["corr_donor"]

    eligible = findall(corr .>= corr_thr)
    isempty(eligible) && error("No eligible RSC blocks at corr_thr=$corr_thr")
    rng = MersenneTwister(seed_sel); shuffle!(rng, eligible)
    start2 = clamp(start, 1, length(eligible))
    blk_list = eligible[start2 : min(length(eligible), start2 + nblk_rsc - 1)]
    isempty(blk_list) && error("Empty blk_list")

    df = DataFrame(
        p=Float64[], blk=Int[], corr=Float64[], method=String[],
        u64_ber=Float64[], u64_psr=Float64[],
        b128_post_ber=Float64[], b128_post_psr=Float64[],
        b128_ch_ber=Float64[], sigma2_hat=Float64[]
    )

    println("==============================================================")
    @printf("RSC DETAIL | methods=%s | using=%d blocks | psweep=%s\n", join(rsc_methods, ","), length(blk_list), string(psweep_rsc))
    println("==============================================================")

    for p in psweep_rsc
        pilots_pos = choose_pilots_bits(128; frac=p)
        for method in rsc_methods
            @printf("\n--- RSC p=%.2f method=%s ---\n", p, method)
            for (ii, b) in enumerate(blk_list)
                y = ComplexF64.(vec(ymat[b, :]))
                u_true = Int.(vec(umat[b, :]))
                b_true = Int.(vec(bmat[b, :]))
                h = ComplexF64.(vec(hmat[b, :]))

                if lowercase(method) == "turboeq"
                    out = decode_turboeq_rsc_bpsk(
                        y, h, u_true, b_true;
                        p=p, turbo_iters=turbo_iters, σ2_init=σ2_init_turbo,
                        eq_σ2_iters=eq_σ2_iters, llr_clip=llr_clip
                    )
                    u_hat = Vector{Int}(out.u64_hat)
                    b_hat_ch   = hard_from_llr(out.llr128_ch)
                    b_hat_post = hard_from_llr(out.llr128_post)

                    push!(df, (p=p, blk=b, corr=Float64(corr[b]), method="TurboEQ",
                               u64_ber=ber(u_hat, u_true), u64_psr=psr_pkt(u_hat, u_true),
                               b128_post_ber=ber(b_hat_post, b_true), b128_post_psr=psr_pkt(b_hat_post, b_true),
                               b128_ch_ber=ber(b_hat_ch, b_true),
                               sigma2_hat=Float64(out.sigma2_hat)))

                elseif lowercase(method) == "lineareq"
                    pilots_bits = isempty(pilots_pos) ? Int[] : Int.(b_true[pilots_pos])
                    out = decode_lineareq_rsc_bpsk(
                        y, h;
                        pilots_pos=pilots_pos,
                        pilots_bits=pilots_bits,
                        M=fir_M, D=fir_D,
                        iters=lin_iters,
                        σ2_init=σ2_init_lin,
                        llr_clip=llr_clip,
                        u64_true=u_true,
                        b128_true=b_true
                    )
                    u_hat = Vector{Int}(out.u64_hat)
                    b_hat_ch   = hard_from_llr(out.llr128_ch)
                    b_hat_post = hard_from_llr(out.llr128_post)
                    push!(df, (p=p, blk=b, corr=Float64(corr[b]), method="LinearEQ",
                               u64_ber=ber(u_hat, u_true), u64_psr=psr_pkt(u_hat, u_true),
                               b128_post_ber=ber(b_hat_post, b_true), b128_post_psr=psr_pkt(b_hat_post, b_true),
                               b128_ch_ber=ber(b_hat_ch, b_true),
                               sigma2_hat=Float64(out.sigma2_hat)))
                else
                    error("Unknown RSC method '$method'")
                end

                if ii == 1 || ii % 100 == 0 || ii == length(blk_list)
                    @printf("  blk %d/%d | u64_psr=%.2f b128_post_ber=%.3f\n",
                            ii, length(blk_list), df.u64_psr[end], df.b128_post_ber[end])
                end
            end
        end
    end

    mkpath(dirname(out_csv))
    CSV.write(out_csv, df)
    println("\nSaved RSC detail → $out_csv")
    return df
end

# ============================================================
# RUN RAW (sweep p): EQ+SPA via p-dependent h refine, plus JSDC + JSDC->SPA
# ============================================================
function run_raw_detail(; raw_path::String,
                        cache_path::String,
                        out_csv::String,
                        nframes_use::Int,
                        psweep_raw::Vector{Float64},
                        npc::Int,
                        T_frame::Int,
                        num_data::Int,
                        num_repeats::Int,
                        h_len::Int,
                        rho_ls::Float64,
                        w_pil::Float64,
                        spa_max_iter::Int,
                        llr_clip::Float64,
                        jsdc_max_iter::Int,
                        jsdc_etaz::Float64,
                        jsdc_lampar::Float64,
                        jsdc_lamprior::Float64,
                        jsdc_lampil::Float64)

    isfile(raw_path)   || error("Missing RAW rec file: $raw_path")
    isfile(cache_path) || error("Missing cache file: $cache_path")

    d = JLD2.load(raw_path)
    all_packets_df = DataFrame(d["all_packets_df"])
    packet_matrix = Matrix(select(all_packets_df, Not(:frame)))
    num_frames = size(packet_matrix, 1)

    dc = JLD2.load(cache_path)
    bestD = Vector{Int}(dc["bestD"])
    length(bestD) == num_frames || error("cache frames mismatch")

    # donor LDPC(64->128)
    codeB, colsB, idrowsB, _ = initcode(64, 128, npc)
    codeB.icols === nothing && (encode(codeB, zeros(Int, 64)); nothing)
    HB = get_H_sparse(codeB)
    parityB = build_parity_indices(HB)

    # rebuild true x and cw (debug/genie generation)
    x_old_mat, cw_true_mat = rebuild_x_old_and_cw_mat(codeB; T_frame=T_frame, num_data=num_data, num_repeats=num_repeats)
    size(x_old_mat, 1) == num_frames || error("x_old_mat rows mismatch vs frames (check num_data*num_repeats)")

    useN = (nframes_use < 0) ? num_frames : min(nframes_use, num_frames)

    df = DataFrame(
        p=Float64[],
        frame=Int[], bestD=Int[], sigma2=Float64[],
        method=String[],
        ber=Float64[], psr_pkt=Float64[], psr64=Float64[],
        sw=Float64[], spa_iters=Int[]
    )

    println("==============================================================")
    @printf("RAW DETAIL | use=%d frames | psweep_raw=%s | EQ+SPA uses p-dependent channel refine (non-oracle)\n",
            useN, string(psweep_raw))
    println("==============================================================")

    for f in 1:useN
        y = extract_symbol_rate(packet_matrix[f, :], T_frame)
        y = shift_left(y, bestD[f])

        cw_true = Int.(vec(cw_true_mat[f, :]))

        # ---- initial DD x0 from y (non-oracle) ----
        x0_bits = [real(y[t]) >= 0 ? 1 : 0 for t in 1:T_frame]
        x0 = ComplexF64.(bpsk_from_bit.(x0_bits))

        # ---- initial h0 from x0 ----
        h0 = ridge_wls_h_from_x(y, x0, h_len, rho_ls, Int[], 0.0)
        σ2_0 = clamp(Float64(mean(abs2, y .- conv_prefix(h0, x0, T_frame))), 1e-6, 10.0)

        # ---- initial equalization using h0 ----
        xeq0_c = eq_lmmse_deconv(y, h0, σ2_0)
        xeq0 = real.(xeq0_c)

        # JSDC warm start/prior from xeq0
        m_init = clamp.(xeq0, -0.999, 0.999)
        z_init = atanh.(m_init)
        L_prior = 2.0 .* z_init
        h_pos = collect(1:h_len)

        for p in psweep_raw
            # ============================================================
            # EQ+SPA: refine h using more pseudo pilots (non-oracle)
            # ============================================================
            pil_pos, pil_bits = pseudo_pilots_from_xeq(xeq0; frac=p)

            # x_ref from decisions; keep as-is, but pilots get higher weight
            xref_bits = [xeq0[t] >= 0 ? 1 : 0 for t in 1:T_frame]
            x_ref = ComplexF64.(bpsk_from_bit.(xref_bits))

            h_ref = ridge_wls_h_from_x(y, x_ref, h_len, rho_ls, pil_pos, w_pil)
            σ2_ref = clamp(Float64(mean(abs2, y .- conv_prefix(h_ref, x_ref, T_frame))), 1e-6, 10.0)

            xeq_c = eq_lmmse_deconv(y, h_ref, σ2_ref)
            xeq = real.(xeq_c)

            L01 = bpsk_L01_from_xeq(xeq, σ2_ref; clipL=llr_clip)
            cw_hat_eq, it_eq = spa_from_llr(codeB, colsB, parityB, L01; max_iter=spa_max_iter)

            ber_eq   = mean(cw_hat_eq .!= cw_true)
            psr_eq   = psr_pkt(cw_hat_eq, cw_true)
            psr64_eq = psr_segments(cw_hat_eq, cw_true; seg=64)
            sw_eq    = syndrome_weight(HB, cw_hat_eq)

            push!(df, (p, f, bestD[f], σ2_ref, "EQ+SPA", ber_eq, psr_eq, psr64_eq, sw_eq, it_eq))

            # ============================================================
            # JSDC: oracle pilots at this p (as your earlier debug)
            # ============================================================
            pilot_pos = choose_pilots_bits(T_frame; frac=p)
            pilot_bpsk = isempty(pilot_pos) ? Float64[] : Float64[bpsk_from_bit(cw_true[j]) for j in pilot_pos]

            xhat_js, m_js = if HAVE_JSDC_BPSK
                xh, _hh, info = LS.jsdc_bpsk_manual(
                    y, codeB, parityB, pilot_pos, pilot_bpsk, h_pos;
                    λ_par=jsdc_lampar,
                    λ_pil=jsdc_lampil,
                    λ_prior=jsdc_lamprior,
                    η_z=jsdc_etaz,
                    η_h=0.0,
                    σ2_data=σ2_ref,
                    max_iter=jsdc_max_iter,
                    h_init=h_ref[1:h_len],
                    z_init=z_init,
                    L_prior=L_prior,
                    verbose=false
                )
                xbits = Int.(xh)
                mvec  = hasproperty(info, :m_final) ? Vector{Float64}(info.m_final) : (2.0 .* Float64.(xbits) .- 1.0)
                (xbits, mvec)
            else
                xh, _hh, info = jsdc_qpsk_manual(
                    y, codeB, parityB, pilot_pos, pilot_bpsk, h_pos;
                    modulation=:bpsk,
                    λ_par=jsdc_lampar,
                    λ_pil=jsdc_lampil,
                    γ_z=5e-3,
                    γ_h=1e-3,
                    η_z=jsdc_etaz,
                    η_h=0.0,
                    max_iter=jsdc_max_iter,
                    h_init=h_ref[1:h_len],
                    z_init=z_init,
                    L_prior=L_prior,
                    λ_prior=jsdc_lamprior,
                    σ2_data=σ2_ref,
                    verbose=false
                )
                xbits = Int.(xh)
                mvec  = hasproperty(info, :m_final) ? Vector{Float64}(info.m_final) : (2.0 .* Float64.(xbits) .- 1.0)
                (xbits, mvec)
            end

            xhat_js = resolve_flip(xhat_js, HB, pilot_pos, pilot_bpsk)

            ber_js   = mean(xhat_js .!= cw_true)
            psr_js   = psr_pkt(xhat_js, cw_true)
            psr64_js = psr_segments(xhat_js, cw_true; seg=64)
            sw_js    = syndrome_weight(HB, xhat_js)

            push!(df, (p, f, bestD[f], σ2_ref, "JSDC", ber_js, psr_js, psr64_js, sw_js, 0))

            # ============================================================
            # JSDC -> SPA finish
            # ============================================================
            L01_js = llr01_from_m(m_js; clip_L=llr_clip)
            cw_hat_js_spa, it_js_spa = spa_from_llr(codeB, colsB, parityB, L01_js; max_iter=spa_max_iter)

            ber_js_spa   = mean(cw_hat_js_spa .!= cw_true)
            psr_js_spa   = psr_pkt(cw_hat_js_spa, cw_true)
            psr64_js_spa = psr_segments(cw_hat_js_spa, cw_true; seg=64)
            sw_js_spa    = syndrome_weight(HB, cw_hat_js_spa)

            push!(df, (p, f, bestD[f], σ2_ref, "JSDC+SPA", ber_js_spa, psr_js_spa, psr64_js_spa, sw_js_spa, it_js_spa))
        end

        if f == 1 || f % 25 == 0 || f == useN
            @printf("  RAW %4d/%d | done\n", f, useN)
        end
    end

    mkpath(dirname(out_csv))
    CSV.write(out_csv, df)
    println("Saved RAW detail → $out_csv")
    return df
end

# ============================================================
# MAIN
# ============================================================
function main()
    # ---- RSC defaults ----
    rsc_path = joinpath(DATA_DIR, "replayswap_bpsk_RSC_64_128_from_realdata_donorLS_h20_rho1e-2.jld2")
    out_rsc  = joinpath(DATA_DIR, "runs", "psr_bpsk_1_2_rsc_detail.csv")
    corr_thr = 0.10
    nblk_rsc = 200
    seed_sel = 12648430
    start_rsc = 1
    psweep_rsc = collect(0.0:0.1:0.5)
    rsc_methods = ["LinearEQ", "TurboEQ"]

    turbo_iters = 3
    σ2_init_turbo = 0.30
    eq_σ2_iters = 1
    llr_clip = 25.0

    fir_M = 15
    fir_D = 7
    lin_iters = 2
    σ2_init_lin = 0.30

    # ---- RAW defaults ----
    raw_path   = joinpath(DATA_DIR, "raw", "logged_packets_and_ytrain.jld2")
    cache_path = joinpath(DATA_DIR, "ls_cache_h20_rho1e-02_bestD.jld2")
    out_raw    = joinpath(DATA_DIR, "runs", "psr_bpsk_1_2_raw_detail.csv")
    nframes_use = 200
    psweep_raw = collect(0.0:0.1:0.5)

    npc = 4
    T_frame = 128
    num_data = 20
    num_repeats = 45
    h_len = 20
    rho_ls = 1e-2
    w_pil = 10.0
    spa_max_iter = 50

    jsdc_max_iter = 200
    jsdc_etaz = 1e-3
    jsdc_lampar = 0.1
    jsdc_lamprior = 0.5
    jsdc_lampil = 20.0

    out_summary = joinpath(DATA_DIR, "runs", "psr_bpsk_1_2_summary_tidy.csv")

    # CLI (minimal)
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a=="--psweep_rsc"; i+=1; psweep_rsc = parse_psweep(ARGS[i])
        elseif a=="--psweep_raw"; i+=1; psweep_raw = parse_psweep(ARGS[i])
        elseif a=="--nframes"; i+=1; nframes_use = parse_int(ARGS[i])
        elseif a=="--nblk_rsc"; i+=1; nblk_rsc = parse_int(ARGS[i])
        elseif a=="--llr_clip"; i+=1; llr_clip = parse_float(ARGS[i])
        elseif a=="--w_pil"; i+=1; w_pil = parse_float(ARGS[i])

        elseif a=="--jsdc_max_iter"; i+=1; jsdc_max_iter = parse_int(ARGS[i])
        elseif a=="--jsdc_etaz"; i+=1; jsdc_etaz = parse_float(ARGS[i])
        elseif a=="--jsdc_lampar"; i+=1; jsdc_lampar = parse_float(ARGS[i])
        elseif a=="--jsdc_lamprior"; i+=1; jsdc_lamprior = parse_float(ARGS[i])
        elseif a=="--jsdc_lampil"; i+=1; jsdc_lampil = parse_float(ARGS[i])
        else
            error("Unknown arg: $a")
        end
        i += 1
    end

    df_rsc = run_rsc_detail(
        rsc_path=rsc_path, out_csv=out_rsc,
        psweep_rsc=psweep_rsc, rsc_methods=rsc_methods,
        corr_thr=corr_thr, nblk_rsc=nblk_rsc, seed_sel=seed_sel, start=start_rsc,
        turbo_iters=turbo_iters, σ2_init_turbo=σ2_init_turbo, eq_σ2_iters=eq_σ2_iters, llr_clip=llr_clip,
        fir_M=fir_M, fir_D=fir_D, lin_iters=lin_iters, σ2_init_lin=σ2_init_lin
    )

    df_raw = run_raw_detail(
        raw_path=raw_path, cache_path=cache_path, out_csv=out_raw,
        nframes_use=nframes_use, psweep_raw=psweep_raw,
        npc=npc, T_frame=T_frame, num_data=num_data, num_repeats=num_repeats,
        h_len=h_len, rho_ls=rho_ls, w_pil=w_pil,
        spa_max_iter=spa_max_iter, llr_clip=llr_clip,
        jsdc_max_iter=jsdc_max_iter, jsdc_etaz=jsdc_etaz,
        jsdc_lampar=jsdc_lampar, jsdc_lamprior=jsdc_lamprior, jsdc_lampil=jsdc_lampil
    )

    rsc_sum = combine(groupby(df_rsc, [:p, :method]),
        :u64_psr       => mean   => :u64_psr,
        :u64_ber       => mean   => :u64_ber,
        :b128_post_psr => mean   => :b128_post_psr,
        :b128_post_ber => mean   => :b128_post_ber,
        :b128_ch_ber   => mean   => :b128_ch_ber,
        :sigma2_hat    => median => :sigma2_p50
    )
    sort!(rsc_sum, [:p, :method])

    raw_sum = combine(groupby(df_raw, [:p, :method]),
        :ber     => mean => :ber,
        :psr_pkt => mean => :psr_pkt,
        :psr64   => mean => :psr64,
        :sw      => mean => :sw
    )
    sort!(raw_sum, [:p, :method])

    rsc_tidy = melt_metrics(rsc_sum;
        domain="RSC", xname="p", xcol=:p, methodcol=:method,
        metrics=[:u64_psr, :u64_ber, :b128_post_psr, :b128_post_ber, :b128_ch_ber, :sigma2_p50]
    )
    raw_tidy = melt_metrics(raw_sum;
        domain="RAW", xname="p", xcol=:p, methodcol=:method,
        metrics=[:psr64, :psr_pkt, :ber, :sw]
    )

    df_summary = vcat(rsc_tidy, raw_tidy)
    mkpath(dirname(out_summary))
    CSV.write(out_summary, df_summary)

    println("\n--- RSC summary ---")
    show(rsc_sum; allrows=true, allcols=true); println()

    println("\n--- RAW summary ---")
    show(raw_sum; allrows=true, allcols=true); println()

    println("\nSaved tidy summary → $out_summary")
    println("Done.")
end

main()
