#!/usr/bin/env julia
# scripts/merged_rsc_psweep_and_raw_pseudopilot_sweep.jl
#
# ONE SCRIPT:
#   (A) RSC replay-swap BPSK RSC(64->128): sweep p, run EQ(linear) + TurboEQ, plot PSR vs p
#   (B) RAW donor LDPC(64->128) BPSK: sweep pseudo-pilot fraction pilot_frac, run EQ+SPA + JSDC, plot vs pilot_frac
#
# Examples:
#   julia --project=. scripts/merged_rsc_psweep_and_raw_pseudopilot_sweep.jl
#   julia --project=. scripts/merged_rsc_psweep_and_raw_pseudopilot_sweep.jl --ps "0,0.1,0.2,0.3,0.4,0.5" --pilot_fracs "0.02,0.05,0.1,0.2"
#
using Random, Printf, Statistics, LinearAlgebra
using JLD2, DataFrames, CSV
using CairoMakie
using SparseArrays

using DSP
using SignalAnalysis

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
const LS = ensure_linksim_loaded!()
using .LS: initcode, encode, get_H_sparse, sum_product_decode, jsdc_qpsk_manual, Code

# RSC decoders
include(joinpath(ROOT, "lib", "TurboEQ_BPSK_RSC64_128.jl"))
using .TurboEQ_BPSK_RSC64_128: decode_turboeq_rsc_bpsk
include(joinpath(ROOT, "lib", "LinearEQ_BPSK_RSC64_128.jl"))
using .LinearEQ_BPSK_RSC64_128: decode_lineareq_rsc_bpsk

# ----------------------------
# Small utils
# ----------------------------
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
function parse_floats(s::String)
    t = replace(strip(s), " " => "")
    isempty(t) && return Float64[]
    return Float64.(parse.(Float64, split(t, ",")))
end

function choose_pilots_bits(n::Int; frac::Float64)
    frac <= 0 && return Int[]
    Np = max(1, round(Int, frac*n))
    posf = collect(range(1, stop=n, length=Np))
    pos = unique!(clamp.(round.(Int, posf), 1, n))
    sort!(pos)
    return pos
end

# ============================================================
# Part A: RSC psweep (EQ + TurboEQ) -> tidy CSV + plot
# ============================================================
function run_psweep_rsc_methods(; rsc_path::String,
                                corr_thr::Float64,
                                nblk::Int,
                                seed_sel::Int,
                                start::Int,
                                ps::Vector{Float64},
                                turbo_iters::Int,
                                σ2_init::Float64,
                                eq_σ2_iters::Int,
                                llr_clip::Float64,
                                fir_M::Int,
                                fir_D::Int,
                                lin_iters::Int,
                                out_csv::String)

    d = JLD2.load(rsc_path)
    ymat = d["y_bpsk_swapped"]
    umat = d["u64_mat"]
    bmat = d["b128_mat"]
    hmat = d["h_blk_mat"]
    corr = d["corr_donor"]

    eligible = findall(corr .>= corr_thr)
    isempty(eligible) && error("No eligible RSC blocks at corr_thr=$corr_thr")
    rng = MersenneTwister(seed_sel)
    shuffle!(rng, eligible)
    start2 = clamp(start, 1, length(eligible))
    blk_list = eligible[start2 : min(length(eligible), start2 + nblk - 1)]
    isempty(blk_list) && error("Empty blk_list")

    df = DataFrame(p=Float64[], blk=Int[], corr=Float64[],
                   method=String[], psr_b128=Float64[])

    println("==============================================================")
    @printf("RSC PSWEEP | blocks=%d/%d corr_thr=%.2f ps=%s\n",
            length(blk_list), length(eligible), corr_thr, string(ps))
    println("==============================================================")

    for p in ps
        pilots_pos = choose_pilots_bits(128; frac=p)

        for b in blk_list
            y = ComplexF64.(vec(ymat[b, :]))
            u_true = Int.(vec(umat[b, :]))
            b_true = Int.(vec(bmat[b, :]))
            h = ComplexF64.(vec(hmat[b, :]))
            pilots_bits = isempty(pilots_pos) ? Int[] : Int.(b_true[pilots_pos])

            # EQ (linear FIR)
            out_eq = decode_lineareq_rsc_bpsk(
                y, h;
                pilots_pos=pilots_pos, pilots_bits=pilots_bits,
                M=fir_M, D=fir_D, iters=lin_iters,
                σ2_init=σ2_init, llr_clip=llr_clip,
                u64_true=u_true, b128_true=b_true
            )
            b_hat_eq = hard_from_llr(out_eq.llr128_post)
            push!(df, (p, b, Float64(corr[b]), "EQ", psr_pkt(b_hat_eq, b_true)))

            # TurboEQ
            out_t = decode_turboeq_rsc_bpsk(
                y, h, u_true, b_true;
                p=p, turbo_iters=turbo_iters, σ2_init=σ2_init,
                eq_σ2_iters=eq_σ2_iters, llr_clip=llr_clip
            )
            b_hat_t = hard_from_llr(out_t.llr128_post)
            push!(df, (p, b, Float64(corr[b]), "TurboEQ", psr_pkt(b_hat_t, b_true)))
        end
    end

    mkpath(dirname(out_csv))
    CSV.write(out_csv, df)
    println("Saved RSC psweep → $out_csv")
    return df
end

function plot_groupbar_psr_p(; in_csv::String, out_png::String, out_pdf::String, title::String)
    df = CSV.read(in_csv, DataFrame)
    df.p = Float64.(df.p)
    df.method = String.(df.method)

    g = combine(groupby(df, [:p, :method]), :psr_b128 => mean => :psr)
    ps = sort(unique(g.p))
    methods = ["EQ", "TurboEQ"]
    methods = [m for m in methods if m in unique(g.method)]

    vals = Dict(m => [begin
        sub = g[(g.p .== p) .& (g.method .== m), :psr]
        isempty(sub) ? NaN : sub[1]
    end for p in ps] for m in methods)

    fig = Figure(size=(980, 480))
    ax = Axis(fig[1,1], title=title, xlabel="Pilot ratio p", ylabel="PSR")
    ylims!(ax, 0, 1)

    xcenters = collect(1:length(ps))
    group_w = 0.80
    bar_w = group_w / max(length(methods), 1)
    palette = [:dodgerblue, :seagreen, :orange, :purple]

    for (k, m) in enumerate(methods)
        offset = -group_w/2 + (k - 0.5)*bar_w
        x = xcenters .+ offset
        y = vals[m]
        barplot!(ax, x, y; width=bar_w*0.95,
                 color=(palette[1 + mod(k-1, length(palette))], 0.85),
                 strokewidth=1.0, strokecolor=:black, label=m)
        for (xi, yi) in zip(x, y)
            isfinite(yi) || continue
            text!(ax, xi, min(yi + 0.03, 0.98),
                  text=@sprintf("%.3f", yi), align=(:center, :bottom), fontsize=11)
        end
    end

    ax.xticks = (xcenters, [@sprintf("%.1f", p) for p in ps])
    axislegend(ax; position=:rb, framevisible=true)

    mkpath(dirname(out_png)); mkpath(dirname(out_pdf))
    save(out_png, fig); save(out_pdf, fig)
    println("Saved plot → $out_png")
    println("Saved plot → $out_pdf")
end

# ============================================================
# Part B: RAW LDPC pseudo-pilot sweep (EQ+SPA + JSDC) -> tidy CSV + plot
#   (NO-GENIE h): FEC-aided DD + sign-fix + pseudo-pilots for JSDC
# ============================================================

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

function resolve_global_flip_by_soft(bits::Vector{Int}, x_ref::Vector{Float64})
    vote = sum(((2 .* bits .- 1) .* x_ref))
    return (vote < 0) ? (1 .- bits) : bits
end

function resolve_global_flip_by_pilots(bits::Vector{Int}, pilot_pos::Vector{Int}, pilot_bpsk::Vector{Float64})
    isempty(pilot_pos) && return bits
    vote = 0.0
    @inbounds for (k, j) in enumerate(pilot_pos)
        vote += (2*bits[j]-1) * pilot_bpsk[k]
    end
    return (vote < 0) ? (1 .- bits) : bits
end

function resolve_global_flip(bits::Vector{Int}, x_ref::Vector{Float64})
    vote = sum(((2 .* bits .- 1) .* x_ref))
    return (vote < 0) ? (1 .- bits) : bits
end

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

function window_h_by_max(h::Vector{ComplexF64}, Ls::Int)
    Ls <= 0 && return copy(h)
    Ls = min(Ls, length(h))
    ℓ0 = argmax(abs.(h))
    a = max(1, ℓ0 - (Ls ÷ 2))
    b = min(length(h), a + Ls - 1)
    a = max(1, b - Ls + 1)
    return h[a:b]
end

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

    return Vector{ComplexF64}(h), Float64(σ2)
end

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

function fec_aided_dd(y::Vector{ComplexF64},
                      codeB::Code,
                      colsB::Vector{Vector{Int}},
                      parityB::Vector{Vector{Int}};
                      h_len::Int=20,
                      rho::Float64=1e-2,
                      dd_iters::Int=2,
                      outer_rounds::Int=3,
                      max_iter_spa::Int=50)

    h, σ2 = estimate_h_dd_bpsk(y; h_len=h_len, rho=rho, iters=dd_iters)
    h_prev = copy(h)

    for _r in 1:outer_rounds
        x_soft_c = eq_lmmse_prefix(y, h, σ2)
        x_eq_raw = real.(x_soft_c)

        cw_hat, _it, sgn_eq, _sw = spa_from_soft_bestsign(codeB, colsB, parityB, x_eq_raw, σ2; max_iter=max_iter_spa)
        x_eq_used = sgn_eq * x_eq_raw

        cw_hat = resolve_global_flip_by_soft(cw_hat, x_eq_used)

        h_new, σ2_new = estimate_h_from_bits(y, cw_hat; h_len=h_len, rho=rho)

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
        λ_par=λ_par, λ_pil=λ_pil,
        γ_z=γ_z, γ_h=γ_h,
        η_z=η_z, η_h=0.0,
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
        λ_par=λ_par, λ_pil=λ_pil,
        γ_z=γ_z, γ_h=γ_h,
        η_z=η_z, η_h=0.0,
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

    return (swB < swA) ? (bB, -1, swB) : (bA, +1, swA)
end

# TRUE donor cw bits for evaluation only
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

function run_raw_pseudopilot_sweep(; rec_path::String,
                                   cache_path::String,
                                   pilot_fracs::Vector{Float64},
                                   start_frame::Int,
                                   nframes_use::Int,
                                   out_csv::String,
                                   # capture constants
                                   npc::Int=4, T_frame::Int=128, num_data::Int=20, num_repeats::Int=45,
                                   # FEC-aided DD
                                   h_len::Int=20, rho_dd::Float64=1e-2, dd_iters::Int=2, outer_rounds::Int=3,
                                   # pseudo-pilot mode
                                   agree_pilots::Bool=true,
                                   λ_pil::Float64=20.0,
                                   # JSDC
                                   jsdc_max_iter::Int=600, jsdc_ηz::Float64=3e-4, jsdc_λpar::Float64=1.0,
                                   jsdc_γz::Float64=5e-3, jsdc_γh::Float64=1e-3, jsdc_λprior::Float64=1.0,
                                   h_jsdc_Ls::Int=7)

    d = JLD2.load(rec_path)
    all_packets_df = DataFrame(d["all_packets_df"])
    packet_matrix = Matrix(select(all_packets_df, Not(:frame)))
    num_frames = size(packet_matrix, 1)

    dc = JLD2.load(cache_path)
    bestD = Vector{Int}(dc["bestD"])
    length(bestD) == num_frames || error("cache frames mismatch")

    codeB, colsB, _idrowsB, _ = initcode(64, 128, npc)
    codeB.icols === nothing && (encode(codeB, zeros(Int, 64)); nothing)
    H = get_H_sparse(codeB)
    parityB = build_parity_indices(H)

    cw_true_mat = rebuild_cw_true_mat(codeB; T_frame=T_frame, num_data=num_data, num_repeats=num_repeats)

    start_frame = clamp(start_frame, 1, num_frames)
    end_frame = (nframes_use < 0) ? num_frames : min(num_frames, start_frame + nframes_use - 1)
    useN = end_frame - start_frame + 1
    @printf("RAW pilot sweep: frames %d:%d (use %d)\n", start_frame, end_frame, useN)

    out = DataFrame(pilot_frac=Float64[], method=String[],
                    ber=Float64[], psr_pkt=Float64[], psr64=Float64[],
                    nframes=Int[], lam_pil=Float64[], agree_pilots=Int[])

    for frac in pilot_fracs
        ber_eqL = Float64[]; psr_eqL = Float64[]; psr64_eqL = Float64[]
        ber_jsL = Float64[]; psr_jsL = Float64[]; psr64_jsL = Float64[]

        for f in start_frame:end_frame
            y = extract_symbol_rate(packet_matrix[f, :], T_frame)
            y = shift_left(y, bestD[f])
            cw_true = Int.(vec(cw_true_mat[f, :]))

            # FEC-aided no-genie channel/noise
            h_dd, σ2 = fec_aided_dd(y, codeB, colsB, parityB; h_len=h_len, rho=rho_dd, dd_iters=dd_iters, outer_rounds=outer_rounds)

            # EQ soft
            x_soft_c = eq_lmmse_prefix(y, h_dd, σ2)
            x_eq_raw = real.(x_soft_c)

            # EQ -> SPA best sign + flip resolve
            xhat_eq, _it_eq, sgn_eq, _sw_eq = spa_from_soft_bestsign(codeB, colsB, parityB, x_eq_raw, σ2; max_iter=50)
            x_eq_used = sgn_eq * x_eq_raw
            xhat_eq = resolve_global_flip_by_soft(xhat_eq, x_eq_used)

            push!(ber_eqL, mean(xhat_eq .!= cw_true))
            push!(psr_eqL, psr_pkt(xhat_eq, cw_true))
            push!(psr64_eqL, psr_segments(xhat_eq, cw_true; seg=64))

            # Pseudo pilots from oriented EQ output
            pilot_pos, pilot_bpsk = agree_pilots ?
                pick_pseudopilots_agree(x_eq_used, xhat_eq; frac=frac) :
                pick_pseudopilots_magonly(x_eq_used; frac=frac)

            # JSDC warm-start from EQ used
            m_init = clamp.(x_eq_used, -0.999, 0.999)
            z_init = atanh.(m_init)
            L_prior = 2.0 .* z_init

            h_for_jsdc = (h_jsdc_Ls > 0) ? window_h_by_max(h_dd, h_jsdc_Ls) : h_dd

            xhat_jsdc, _sgn_jsdc, _sw = jsdc_bestsign(
                y, codeB, H, parityB, h_for_jsdc, z_init, L_prior, σ2, pilot_pos, pilot_bpsk;
                λ_par=jsdc_λpar, λ_pil=λ_pil, λ_prior=jsdc_λprior,
                η_z=jsdc_ηz, max_iter=jsdc_max_iter, γ_z=jsdc_γz, γ_h=jsdc_γh
            )

            # flip-resolve (pilots then soft)
            x_after_pil = resolve_global_flip_by_pilots(xhat_jsdc, pilot_pos, pilot_bpsk)
            x_after_eq  = resolve_global_flip(x_after_pil, x_eq_used)
            xhat_jsdc   = x_after_eq

            push!(ber_jsL, mean(xhat_jsdc .!= cw_true))
            push!(psr_jsL, psr_pkt(xhat_jsdc, cw_true))
            push!(psr64_jsL, psr_segments(xhat_jsdc, cw_true; seg=64))
        end

        push!(out, (frac, "EQ+SPA",  mean(ber_eqL), mean(psr_eqL), mean(psr64_eqL), useN, λ_pil, agree_pilots ? 1 : 0))
        push!(out, (frac, "JSDC",    mean(ber_jsL), mean(psr_jsL), mean(psr64_jsL), useN, λ_pil, agree_pilots ? 1 : 0))

        @printf("RAW frac=%.3f | EQ: PSR64=%.3f BER=%.3f | JSDC: PSR64=%.3f BER=%.3f\n",
                frac, mean(psr64_eqL), mean(ber_eqL), mean(psr64_jsL), mean(ber_jsL))
    end

    mkpath(dirname(out_csv))
    CSV.write(out_csv, out)
    println("Saved RAW pilot sweep → $out_csv")
    return out
end

function plot_groupbar_raw(; in_csv::String, out_png::String, out_pdf::String,
                           metric::Symbol=:psr64, title::String="RAW LDPC vs pseudo-pilot fraction")
    df = CSV.read(in_csv, DataFrame)
    df.pilot_frac = Float64.(df.pilot_frac)
    df.method = String.(df.method)

    fracs = sort(unique(df.pilot_frac))
    methods = ["EQ+SPA", "JSDC"]
    methods = [m for m in methods if m in unique(df.method)]

    vals = Dict(m => [begin
        sub = df[(df.pilot_frac .== f) .& (df.method .== m), metric]
        isempty(sub) ? NaN : Float64(sub[1])
    end for f in fracs] for m in methods)

    fig = Figure(size=(980, 480))
    ylabel = metric == :ber ? "BER" : metric == :psr_pkt ? "PSR(pkt)" : "PSR64"
    ax = Axis(fig[1,1], title=title, xlabel="pseudo-pilot fraction", ylabel=ylabel)
    metric == :ber ? ylims!(ax, 0, max(0.5, 1.05*maximum(skipmissing(vcat([vals[m] for m in methods]...))))) : ylims!(ax, 0, 1)

    xcenters = collect(1:length(fracs))
    group_w = 0.80
    bar_w = group_w / max(length(methods), 1)
    palette = [:dodgerblue, :orange, :seagreen, :purple]

    for (k, m) in enumerate(methods)
        offset = -group_w/2 + (k - 0.5)*bar_w
        x = xcenters .+ offset
        y = vals[m]
        barplot!(ax, x, y; width=bar_w*0.95,
                 color=(palette[1 + mod(k-1, length(palette))], 0.85),
                 strokewidth=1.0, strokecolor=:black, label=m)
        for (xi, yi) in zip(x, y)
            isfinite(yi) || continue
            metric == :ber || text!(ax, xi, min(yi + 0.03, 0.98),
                                    text=@sprintf("%.3f", yi), align=(:center, :bottom), fontsize=11)
        end
    end

    ax.xticks = (xcenters, [@sprintf("%.3f", f) for f in fracs])
    axislegend(ax; position=:rb, framevisible=true)

    mkpath(dirname(out_png)); mkpath(dirname(out_pdf))
    save(out_png, fig); save(out_pdf, fig)
    println("Saved plot → $out_png")
    println("Saved plot → $out_pdf")
end
function build_combined_summary(; rsc_csv::String, raw_csv::String,
                                raw_metric::Symbol=:psr64,
                                out_csv::String)

    # -------- RSC: tidy -> summarize per p,method --------
    df_r = CSV.read(rsc_csv, DataFrame)
    df_r.p = Float64.(df_r.p)
    df_r.method = String.(df_r.method)

    # RSC currently stores psr_b128 (packet PSR). If you want PSR64 for RSC too,
    # you’d need to compute it in run_psweep_rsc_methods. For now, we map:
    # metric = psr_b128
    gr = combine(groupby(df_r, [:p, :method]),
                 :psr_b128 => mean => :metric)

    rsc_out = DataFrame(
        xkey = ["RSC p=" * @sprintf("%.2f", p) for p in gr.p],
        xval = Float64.(gr.p),
        domain = fill("RSC", nrow(gr)),
        method = gr.method,
        metric = Float64.(gr.metric),
    )

    # -------- RAW: already summarized per pilot_frac,method --------
    df_w = CSV.read(raw_csv, DataFrame)
    df_w.pilot_frac = Float64.(df_w.pilot_frac)
    df_w.method = String.(df_w.method)

    # pick metric column from RAW
    cols = propertynames(df_w)  # Symbols
    cols = propertynames(df_w)  # Symbols
    @assert raw_metric in cols "RAW metric $(raw_metric) not in CSV columns: $(collect(cols))"



    raw_out = DataFrame(
        xkey = ["RAW f=" * @sprintf("%.3f", f) for f in df_w.pilot_frac],
        xval = Float64.(df_w.pilot_frac),
        domain = fill("RAW", nrow(df_w)),
        method = df_w.method,
        metric = Float64.(df_w[:, raw_metric]),
    )

    out = vcat(rsc_out, raw_out)

    mkpath(dirname(out_csv))
    CSV.write(out_csv, out)
    println("Saved combined summary → $out_csv")
    return out
end

function plot_groupbar_combined(; in_csv::String, out_png::String, out_pdf::String,
                                title::String="Combined results",
                                ylabel::String="Metric",
                                methods_order::Vector{String}=["TurboEQ","EQ","EQ+SPA","JSDC"])

    df = CSV.read(in_csv, DataFrame)
    df.method = String.(df.method)
    df.xkey = String.(df.xkey)

    # keep only methods that exist
    methods = [m for m in methods_order if m in unique(df.method)]
    xkeys = unique(df.xkey)

    # keep xkeys in a nice order: all RSC (sorted by xval), then RAW (sorted by xval)
    subr = df[df.domain .== "RSC", :]
    subw = df[df.domain .== "RAW", :]
    r_keys = [k for k in subr[sortperm(subr.xval), :xkey] if k in xkeys]
    w_keys = [k for k in subw[sortperm(subw.xval), :xkey] if k in xkeys]
    xkeys_ord = unique(vcat(r_keys, w_keys))

    # values[m][i] = metric for method m at xkey i
    vals = Dict(m => [begin
        sub = df[(df.xkey .== xk) .& (df.method .== m), :metric]
        isempty(sub) ? NaN : Float64(sub[1])
    end for xk in xkeys_ord] for m in methods)

    fig = Figure(size=(1200, 520))
    ax = Axis(fig[1,1], title=title, xlabel="Sweep point", ylabel=ylabel)
    ylims!(ax, 0, 1)  # good for PSR-style metrics

    xcenters = collect(1:length(xkeys_ord))
    group_w = 0.85
    bar_w = group_w / max(length(methods), 1)
    palette = [:dodgerblue, :seagreen, :orange, :purple, :crimson, :gray40]

    for (k, m) in enumerate(methods)
        offset = -group_w/2 + (k - 0.5)*bar_w
        x = xcenters .+ offset
        y = vals[m]
        barplot!(ax, x, y; width=bar_w*0.95,
                 color=(palette[1 + mod(k-1, length(palette))], 0.85),
                 strokewidth=1.0, strokecolor=:black, label=m)
        for (xi, yi) in zip(x, y)
            isfinite(yi) || continue
            text!(ax, xi, min(yi + 0.03, 0.98),
                  text=@sprintf("%.3f", yi), align=(:center, :bottom), fontsize=10)
        end
    end

    ax.xticks = (xcenters, xkeys_ord)
    ax.xticklabelrotation = π/6
    axislegend(ax; position=:rb, framevisible=true)

    mkpath(dirname(out_png)); mkpath(dirname(out_pdf))
    save(out_png, fig); save(out_pdf, fig)
    println("Saved combined plot → $out_png")
    println("Saved combined plot → $out_pdf")
end

# ============================================================
# MAIN
# ============================================================
function main()
    # -------- RSC defaults --------
    rsc_path = joinpath(DATA_DIR, "replayswap_bpsk_RSC_64_128_from_realdata_donorLS_h20_rho1e-2.jld2")
    corr_thr_rsc = 0.10
    nblk_rsc = 200
    seed_sel_rsc = 12648430
    start_rsc = 1
    ps = [0.0,0.1,0.2,0.3,0.4,0.5]

    turbo_iters = 2
    σ2_init = 0.30
    eq_σ2_iters = 1
    llr_clip = 25.0

    fir_M = 15
    fir_D = 7
    lin_iters = 2

    out_rsc_csv = joinpath(DATA_DIR, "runs", "psweep_rsc_b128_tidy.csv")
    out_rsc_png = joinpath(DATA_DIR, "plots", "rsc_psrbar.png")
    out_rsc_pdf = joinpath(DATA_DIR, "plots", "rsc_psrbar.pdf")

    # -------- RAW defaults --------
    rec_path   = joinpath(DATA_DIR, "raw", "logged_packets_and_ytrain.jld2")
    cache_path = joinpath(DATA_DIR, "ls_cache_h20_rho1e-02_bestD.jld2")
    pilot_fracs = [0.0, 0.1, 0.20, 0.30, 0.4, 0.5]
    start_raw = 1
    nframes_raw = 200
    agree_pilots = true
    lam_pil = 20.0
    raw_metric = "psr64"  # psr64 | psr_pkt | ber

    out_raw_csv = joinpath(DATA_DIR, "runs", "raw_pseudopilot_sweep_tidy.csv")
    out_raw_png = joinpath(DATA_DIR, "plots", "raw_pseudopilot_sweep.png")
    out_raw_pdf = joinpath(DATA_DIR, "plots", "raw_pseudopilot_sweep.pdf")
# ---- COMBINED ----
out_comb_csv = joinpath(DATA_DIR, "runs", "combined_methods_summary.csv")
out_comb_png = joinpath(DATA_DIR, "plots", "combined_methods.png")
out_comb_pdf = joinpath(DATA_DIR, "plots", "combined_methods.pdf")

metric_sym = raw_metric == "ber" ? :ber : raw_metric == "psr_pkt" ? :psr_pkt : :psr64

build_combined_summary(; rsc_csv=out_rsc_csv, raw_csv=out_raw_csv,
                       raw_metric=metric_sym,
                       out_csv=out_comb_csv)

plot_groupbar_combined(; in_csv=out_comb_csv, out_png=out_comb_png, out_pdf=out_comb_pdf,
                       title="TurboEQ vs EQ+SPA vs JSDC (combined RSC+RAW)",
                       ylabel = (metric_sym==:ber ? "BER" : metric_sym==:psr_pkt ? "PSR(pkt)" : "PSR64/PSR(pkt)"))

    # CLI
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a=="--ps"; i+=1; ps = parse_floats(ARGS[i])
        elseif a=="--pilot_fracs"; i+=1; pilot_fracs = parse_floats(ARGS[i])
        elseif a=="--nblk_rsc"; i+=1; nblk_rsc = parse_int(ARGS[i])
        elseif a=="--corr_rsc"; i+=1; corr_thr_rsc = parse(Float64, ARGS[i])

        elseif a=="--nframes_raw"; i+=1; nframes_raw = parse_int(ARGS[i])
        elseif a=="--start_raw"; i+=1; start_raw = parse_int(ARGS[i])
        elseif a=="--agree_pilots"; i+=1; agree_pilots = (parse_int(ARGS[i]) != 0)
        elseif a=="--lam_pil"; i+=1; lam_pil = parse(Float64, ARGS[i])
        elseif a=="--raw_metric"; i+=1; raw_metric = ARGS[i]

        elseif a=="--out_rsc_csv"; i+=1; out_rsc_csv = ARGS[i]
        elseif a=="--out_raw_csv"; i+=1; out_raw_csv = ARGS[i]
        elseif a=="--help" || a=="-h"
            println("""
Usage:
  julia --project=. scripts/merged_rsc_psweep_and_raw_pseudopilot_sweep.jl [args]

RSC:
  --ps "0,0.1,0.2,0.3,0.4,0.5"
  --corr_rsc <float>
  --nblk_rsc <int>

RAW:
  --pilot_fracs "0.02,0.05,0.1,0.2"
  --start_raw <int>
  --nframes_raw <int>      (-1 = all from start)
  --agree_pilots 0|1       (1 = agree-with-EQ, 0 = mag-only)
  --lam_pil <float>
  --raw_metric psr64|psr_pkt|ber

Outputs:
  --out_rsc_csv <path>
  --out_raw_csv <path>
""")
            return
        else
            error("Unknown arg: $a")
        end
        i += 1
    end

    # ---- RSC ----
    run_psweep_rsc_methods(; rsc_path, corr_thr=corr_thr_rsc, nblk=nblk_rsc,
                           seed_sel=seed_sel_rsc, start=start_rsc, ps,
                           turbo_iters, σ2_init, eq_σ2_iters, llr_clip,
                           fir_M, fir_D, lin_iters, out_csv=out_rsc_csv)

    plot_groupbar_psr_p(; in_csv=out_rsc_csv, out_png=out_rsc_png, out_pdf=out_rsc_pdf,
                        title="RSC b128 PSR vs Pilot Ratio")

    # ---- RAW ----
    isfile(rec_path) || error("Missing RAW rec file: $rec_path")
    isfile(cache_path) || error("Missing cache file: $cache_path")

    run_raw_pseudopilot_sweep(; rec_path, cache_path,
                              pilot_fracs, start_frame=start_raw, nframes_use=nframes_raw,
                              out_csv=out_raw_csv,
                              agree_pilots=agree_pilots, λ_pil=lam_pil)

    metric_sym = raw_metric == "ber" ? :ber : raw_metric == "psr_pkt" ? :psr_pkt : :psr64
    plot_groupbar_raw(; in_csv=out_raw_csv, out_png=out_raw_png, out_pdf=out_raw_pdf,
                      metric=metric_sym,
                      title="RAW LDPC vs pseudo-pilot fraction (agree_pilots=$(agree_pilots ? 1 : 0), λ_pil=$(lam_pil))")
end

main()
