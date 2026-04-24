#!/usr/bin/env julia
# scripts/sweep_rsc_turbo_and_rawldpc_eqspa_jsdc.jl
#
# Sweeps the "compare_rsc_turbo_and_rawldpc_eqspa_jsdc" idea.
#
# RSC replay-swap (BPSK, RSC 64->128):
#   - Turbo decode = EQ -> LLR -> (optional pilot clamp) -> BCJR -> posterior mean -> refine σ² -> repeat
#   - Sweeps: p (pilot fraction), corr_thr, turbo_iters, σ2_init
#
# RAW donor LDPC (BPSK, LDPC 64->128):
#   - Methods: EQ+SPA and JSDC
#   - Runs once per sweep row (or you can also sweep jsdc knobs)
#
# Output:
#   data/runs/sweep_rsc_turbo_and_rawldpc_eqspa_jsdc.csv
#
# Examples:
#   julia --project=. scripts/sweep_rsc_turbo_and_rawldpc_eqspa_jsdc.jl
#
#   julia --project=. scripts/sweep_rsc_turbo_and_rawldpc_eqspa_jsdc.jl \
#     --ps "0,0.05,0.10,0.20" --corrs "0.10,0.40" --turbo_iters "2,4,6" --nblk_rsc 225
#
#   julia --project=. scripts/sweep_rsc_turbo_and_rawldpc_eqspa_jsdc.jl \
#     --do_raw 0   # only sweep RSC

using Random, Printf, Statistics, LinearAlgebra
using JLD2, DataFrames, CSV
using SparseArrays

# RAW donor helpers
using DSP
using SignalAnalysis

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
const LS = ensure_linksim_loaded!()
using .LS: initcode, encode, get_H_sparse, sum_product_decode, jsdc_qpsk_manual, Code

# ----------------------------
# parsing helpers
# ----------------------------
function parse_floats(s::String)
    ss = replace(strip(s), " " => "")
    isempty(ss) && return Float64[]
    return Float64.(parse.(Float64, split(ss, ",")))
end
function parse_ints(s::String)
    ss = replace(strip(s), " " => "")
    isempty(ss) && return Int[]
    return Int.(parse.(Int, split(ss, ",")))
end

# ----------------------------
# common small utils
# ----------------------------
@inline hard_from_llr(L::AbstractVector{<:Real}) = Int.(L .< 0)
@inline ber(a::AbstractVector{Int}, b::AbstractVector{Int}) = mean(a .!= b)
@inline psr_pkt(a::AbstractVector{Int}, b::AbstractVector{Int}) = all(a .== b) ? 1.0 : 0.0

@inline function conv_prefix(h::Vector{ComplexF64}, x::Vector{ComplexF64}, T::Int)
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

@inline function convmtx_prefix(h::Vector{ComplexF64}, T::Int)
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

function eq_lmmse_deconv(y::Vector{ComplexF64}, h::Vector{ComplexF64}, σ2::Float64)
    T = length(y)
    Hc = convmtx_prefix(h, T)
    M = Hc' * Hc
    @inbounds for i in 1:T
        M[i,i] += max(σ2, 1e-12)
    end
    rhs = Hc' * y
    return M \ rhs
end

function eq_lmmse_with_sigma2(y::Vector{ComplexF64}, h::Vector{ComplexF64};
                              σ2_init::Float64=0.30, iters::Int=2)
    σ2 = max(σ2_init, 1e-6)
    x = eq_lmmse_deconv(y, h, σ2)
    for _ in 1:iters
        yhat = conv_prefix(h, x, length(y))
        σ2 = clamp(Float64(mean(abs2, y .- yhat)), 1e-6, 10.0)
        x = eq_lmmse_deconv(y, h, σ2)
    end
    return x, σ2
end

@inline function bpsk_llr_logP0P1(x_soft::Vector{Float64}, σ2::Float64; clip::Float64=25.0)
    c = -2.0 / max(σ2, 1e-12)
    L = c .* x_soft
    return clamp.(L, -clip, clip)
end

@inline bpsk_mean_from_L01(L01::Vector{Float64}) = -tanh.(0.5 .* L01)

# evenly spaced pilot bit positions in 1..n
function choose_pilots_bits(n::Int; frac::Float64)
    frac <= 0 && return Int[]
    Np = max(1, round(Int, frac*n))
    posf = collect(range(1, stop=n, length=Np))
    pos = unique!(clamp.(round.(Int, posf), 1, n))
    sort!(pos)
    return pos
end

@inline function clamp_pilots_L01!(L01::Vector{Float64}, btrue::Vector{Int}, pos::Vector{Int}; clampL::Float64=25.0)
    @inbounds for p in pos
        L01[p] = (btrue[p] == 0) ? clampL : -clampL
    end
    return L01
end

# ----------------------------
# RSC BCJR (log domain)
# ----------------------------
@inline function lse2(a::Float64, b::Float64)
    if a == -Inf; return b; end
    if b == -Inf; return a; end
    m = max(a,b)
    return m + log(exp(a-m) + exp(b-m))
end

const NST = 4
const next_state = Array{Int}(undef, NST, 2)
const parity_bit = Array{Int}(undef, NST, 2)

function __init_trellis__()
    for s in 0:3
        s1 = (s >> 1) & 1
        s2 = s & 1
        for u in 0:1
            f = u ⊻ s1 ⊻ s2
            p = f ⊻ s2
            ns1 = f
            ns2 = s1
            ns = (ns1 << 1) | ns2
            next_state[s+1, u+1] = ns
            parity_bit[s+1, u+1] = p
        end
    end
end
__init_trellis__()

function bcjr_rsc(Lsys::Vector{Float64}, Lpar::Vector{Float64}, La::Vector{Float64})
    N = length(Lsys)
    @assert length(Lpar) == N
    @assert length(La) == N

    α = fill(-Inf, N+1, NST)
    β = fill(-Inf, N+1, NST)
    α[1, 1] = 0.0
    @inbounds for s in 1:NST
        β[N+1, s] = 0.0
    end

    @inbounds for t in 1:N
        for s in 0:3
            a = α[t, s+1]
            a == -Inf && continue
            for u in 0:1
                ns = next_state[s+1, u+1]
                p  = parity_bit[s+1, u+1]
                g = 0.5 * ( (1 - 2u) * (La[t] + Lsys[t]) + (1 - 2p) * Lpar[t] )
                α[t+1, ns+1] = lse2(α[t+1, ns+1], a + g)
            end
        end
        c = maximum(@view α[t+1, :]); isfinite(c) && (α[t+1, :] .-= c)
    end

    @inbounds for t in N:-1:1
        for s in 0:3
            acc = -Inf
            for u in 0:1
                ns = next_state[s+1, u+1]
                p  = parity_bit[s+1, u+1]
                g = 0.5 * ( (1 - 2u) * (La[t] + Lsys[t]) + (1 - 2p) * Lpar[t] )
                acc = lse2(acc, g + β[t+1, ns+1])
            end
            β[t, s+1] = acc
        end
        c = maximum(@view β[t, :]); isfinite(c) && (β[t, :] .-= c)
    end

    Lu_post = Vector{Float64}(undef, N)
    Lpar_post = Vector{Float64}(undef, N)

    @inbounds for t in 1:N
        num_u0 = -Inf; num_u1 = -Inf
        num_p0 = -Inf; num_p1 = -Inf
        for s in 0:3
            a = α[t, s+1]
            a == -Inf && continue
            for u in 0:1
                ns = next_state[s+1, u+1]
                p  = parity_bit[s+1, u+1]
                g = 0.5 * ( (1 - 2u) * (La[t] + Lsys[t]) + (1 - 2p) * Lpar[t] )
                val = a + g + β[t+1, ns+1]
                if u == 0
                    num_u0 = lse2(num_u0, val)
                else
                    num_u1 = lse2(num_u1, val)
                end
                if p == 0
                    num_p0 = lse2(num_p0, val)
                else
                    num_p1 = lse2(num_p1, val)
                end
            end
        end
        Lu_post[t]   = num_u0 - num_u1
        Lpar_post[t] = num_p0 - num_p1
    end

    return Lu_post, Lpar_post
end

# ----------------------------
# RSC turbo decode
# ----------------------------
function decode_rsc_turbo(y::Vector{ComplexF64}, h::Vector{ComplexF64},
                          b128_true::Vector{Int};
                          p::Float64,
                          turbo_iters::Int,
                          σ2_init::Float64,
                          eq_σ2_iters::Int,
                          llr_clip::Float64)

    σ2 = σ2_init
    L01_last = zeros(Float64, 128)
    u_hat = zeros(Int, 64)

    pilot_pos = choose_pilots_bits(128; frac=p)

    for _ in 1:max(1, turbo_iters)
        x_soft_c, σ2_hat = eq_lmmse_with_sigma2(y, h; σ2_init=σ2, iters=eq_σ2_iters)
        x_soft = real.(x_soft_c)

        L01 = bpsk_llr_logP0P1(x_soft, σ2_hat; clip=llr_clip)
        if !isempty(pilot_pos)
            clamp_pilots_L01!(L01, b128_true, pilot_pos; clampL=llr_clip)
        end

        Lsys = L01[1:2:end]
        Lpar = L01[2:2:end]
        Lu_post, Lpar_post = bcjr_rsc(Lsys, Lpar, zeros(Float64, 64))
        u_hat = hard_from_llr(Lu_post)

        # reconstruct codeword posterior means to refine σ²
        # Use Lu_post / Lpar_post as code-bit posteriors (reasonable for turbo loop)
        Lcode_post = Vector{Float64}(undef, 128)
        @inbounds for t in 1:64
            Lcode_post[2t-1] = Lu_post[t]
            Lcode_post[2t]   = Lpar_post[t]
        end
        xmean = bpsk_mean_from_L01(Lcode_post)
        yhat = conv_prefix(h, ComplexF64.(xmean), length(y))
        σ2 = clamp(Float64(mean(abs2, y .- yhat)), 1e-6, 10.0)

        L01_last = L01
    end

    return u_hat, L01_last, σ2
end

# ============================================================
# RAW donor LDPC decode bits (reuse your logic)
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
        d_test = Int.((bseq .+ 1) ./ 2)
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

function run_raw_ldpc(raw_path::String, cache_path::String;
                      nframes_use::Int,
                      npc::Int, T_frame::Int, num_data::Int, num_repeats::Int,
                      h_len::Int, rho_ls::Float64,
                      jsdc_max_iter::Int, jsdc_ηz::Float64, jsdc_λpar::Float64, jsdc_λprior::Float64,
                      jsdc_γz::Float64, jsdc_γh::Float64)

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

    ber_eqspa_list = Float64[]
    ber_jsdc_list  = Float64[]

    for f in 1:useN
        y = extract_symbol_rate(packet_matrix[f, :], T_frame)
        y = shift_left(y, bestD[f])

        x_true  = vec(x_old_mat[f, :])
        cw_true = Int.(vec(cw_true_mat[f, :]))

        h = ridge_ls_h(x_true, y, h_len, rho_ls)

        yhat = conv_prefix(h, x_true, T_frame)
        σ2 = Float64(mean(abs2, y .- yhat))

        # EQ+SPA
        Hc = convmtx_prefix(h, T_frame)
        A = Hc' * Hc
        @inbounds for ii in 1:T_frame
            A[ii,ii] += max(σ2, 1e-6)
        end
        rhs = Hc' * y
        x_lmmse = A \ rhs
        x_eq = real.(x_lmmse)
        cw_hat_eq, _it_spa = spa_from_soft(codeB, colsB, parityB, x_eq, σ2; max_iter=50)
        push!(ber_eqspa_list, mean(cw_hat_eq .!= cw_true))

        # JSDC
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
        push!(ber_jsdc_list, mean(Int.(xhat_jsdc) .!= cw_true))
    end

    return mean(ber_eqspa_list), mean(ber_jsdc_list), useN
end

# ============================================================
# main sweep
# ============================================================
function main()
    # paths
    rsc_path   = joinpath(DATA_DIR, "replayswap_bpsk_RSC_64_128_from_realdata_donorLS_h20_rho1e-2.jld2")
    raw_path   = joinpath(DATA_DIR, "raw", "logged_packets_and_ytrain.jld2")
    cache_path = joinpath(DATA_DIR, "ls_cache_h20_rho1e-02_bestD.jld2")
    outcsv     = joinpath(DATA_DIR, "runs", "sweep_rsc_turbo_and_rawldpc_eqspa_jsdc.csv")

    # sweep lists (defaults)
    ps        = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    corrs     = [0.10]
    turbo_itL = [2, 4]
    σ2_inits  = [0.30]

    # selection
    nblk_rsc  = 225
    seed_sel  = 12648430
    start     = 1

    # RSC decode knobs
    eq_σ2_iters = 1
    llr_clip    = 25.0

    # raw config
    do_raw      = true
    nframes_use = 200

    # capture generation constants
    npc = 4
    T_frame = 128
    num_data = 20
    num_repeats = 45
    h_len = 20
    rho_ls = 1e-2

    # JSDC defaults (can sweep too, but keeping simple)
    jsdc_max_iter = 200
    jsdc_ηz = 1e-3
    jsdc_λpar = 0.1
    jsdc_λprior = 0.5
    jsdc_γz = 5e-3
    jsdc_γh = 1e-3

    # CLI
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a=="--rsc"; i+=1; rsc_path = ARGS[i]
        elseif a=="--raw"; i+=1; raw_path = ARGS[i]
        elseif a=="--cache"; i+=1; cache_path = ARGS[i]
        elseif a=="--outcsv"; i+=1; outcsv = ARGS[i]

        elseif a=="--ps"; i+=1; ps = parse_floats(ARGS[i])
        elseif a=="--corrs"; i+=1; corrs = parse_floats(ARGS[i])
        elseif a=="--turbo_iters"; i+=1; turbo_itL = parse_ints(ARGS[i])
        elseif a=="--σ2_inits"; i+=1; σ2_inits = parse_floats(ARGS[i])

        elseif a=="--nblk_rsc"; i+=1; nblk_rsc = parse(Int, ARGS[i])
        elseif a=="--seed_sel"; i+=1; seed_sel = parse(Int, ARGS[i])
        elseif a=="--start"; i+=1; start = parse(Int, ARGS[i])

        elseif a=="--do_raw"; i+=1; do_raw = (parse(Int, ARGS[i]) != 0)
        elseif a=="--nframes"; i+=1; nframes_use = parse(Int, ARGS[i])

        elseif a=="--jsdc_max_iter"; i+=1; jsdc_max_iter = parse(Int, ARGS[i])
        elseif a=="--jsdc_etaz"; i+=1; jsdc_ηz = parse(Float64, ARGS[i])
        elseif a=="--jsdc_lampar"; i+=1; jsdc_λpar = parse(Float64, ARGS[i])
        elseif a=="--jsdc_lamprior"; i+=1; jsdc_λprior = parse(Float64, ARGS[i])
        else
            error("Unknown arg: $a")
        end
        i += 1
    end

    isfile(rsc_path) || error("Missing RSC dataset: $rsc_path")
    isfile(raw_path) || !do_raw || error("Missing RAW dataset: $raw_path")
    isfile(cache_path) || !do_raw || error("Missing cache: $cache_path")

    mkpath(dirname(outcsv))

    # Load RSC dataset once
    dr = JLD2.load(rsc_path)
    ymat = dr["y_bpsk_swapped"]
    umat = dr["u64_mat"]
    bmat = dr["b128_mat"]
    hmat = dr["h_blk_mat"]
    corr = dr["corr_donor"]

    nblk_all = size(ymat, 1)

    # RAW results (computed once per run, not per RSC sweep) — but we’ll log them in every row.
    raw_ber_eqspa = NaN
    raw_ber_jsdc  = NaN
    raw_usedN     = 0
    if do_raw
        @printf("Running RAW donor once (nframes=%d)...\n", nframes_use)
        raw_ber_eqspa, raw_ber_jsdc, raw_usedN = run_raw_ldpc(
            raw_path, cache_path;
            nframes_use=nframes_use,
            npc=npc, T_frame=T_frame, num_data=num_data, num_repeats=num_repeats,
            h_len=h_len, rho_ls=rho_ls,
            jsdc_max_iter=jsdc_max_iter, jsdc_ηz=jsdc_ηz, jsdc_λpar=jsdc_λpar, jsdc_λprior=jsdc_λprior,
            jsdc_γz=jsdc_γz, jsdc_γh=jsdc_γh
        )
        @printf("RAW donor mean BER(cw128): EQ+SPA=%.4f  JSDC=%.4f (used %d frames)\n\n",
                raw_ber_eqspa, raw_ber_jsdc, raw_usedN)
    end

    # Output table
    out = DataFrame(
        corr_thr=Float64[], p=Float64[], turbo_iters=Int[], σ2_init=Float64[],
        nblk_used=Int[],
        rsc_psr_u64=Float64[], rsc_ber_u64=Float64[],
        rsc_psr_b128=Float64[], rsc_ber_b128=Float64[],
        raw_ber_eqspa=Float64[], raw_ber_jsdc=Float64[], raw_frames=Int[]
    )

    println("==============================================================")
    @printf("SWEEP RSC Turbo: | ps=%s | corrs=%s | turbo_iters=%s | σ2_inits=%s\n",
            string(ps), string(corrs), string(turbo_itL), string(σ2_inits))
    println("==============================================================")

    for corr_thr in corrs
        eligible = findall(corr .>= corr_thr)
        isempty(eligible) && (@printf("corr_thr=%.2f -> no eligible blocks, skipping\n", corr_thr); continue)
        rng = MersenneTwister(seed_sel)
        shuffle!(rng, eligible)

        start2 = clamp(start, 1, length(eligible))
        blk_list = eligible[start2 : min(length(eligible), start2 + nblk_rsc - 1)]
        isempty(blk_list) && (@printf("corr_thr=%.2f -> empty blk_list after start, skipping\n", corr_thr); continue)

        for p in ps, itT in turbo_itL, σ2_init in σ2_inits
            psr_u = Float64[]
            ber_u = Float64[]
            psr_b = Float64[]
            ber_b = Float64[]

            for b in blk_list
                y = ComplexF64.(vec(ymat[b, :]))
                u_true = Int.(vec(umat[b, :]))
                b_true = Int.(vec(bmat[b, :]))
                hfull  = ComplexF64.(vec(hmat[b, :]))
                Lh = min(length(hfull), length(y))
                h_use = hfull[1:Lh]

                u_hat, L01, _σ2fin = decode_rsc_turbo(
                    y, h_use, b_true;
                    p=p, turbo_iters=itT, σ2_init=σ2_init,
                    eq_σ2_iters=eq_σ2_iters, llr_clip=llr_clip
                )
                b_hat = hard_from_llr(L01)

                push!(psr_u, psr_pkt(u_hat, u_true))
                push!(ber_u, ber(u_hat, u_true))
                push!(psr_b, psr_pkt(b_hat, b_true))
                push!(ber_b, ber(b_hat, b_true))
            end

            push!(out, (
                corr_thr=Float64(corr_thr), p=Float64(p), turbo_iters=Int(itT), σ2_init=Float64(σ2_init),
                nblk_used=length(blk_list),
                rsc_psr_u64=mean(psr_u), rsc_ber_u64=mean(ber_u),
                rsc_psr_b128=mean(psr_b), rsc_ber_b128=mean(ber_b),
                raw_ber_eqspa=Float64(raw_ber_eqspa), raw_ber_jsdc=Float64(raw_ber_jsdc), raw_frames=Int(raw_usedN)
            ))

            @printf("corr=%.2f p=%.2f it=%d σ2=%.2f | RSC u64 PSR=%.3f BER=%.3f | b128 PSR=%.3f BER=%.3f\n",
                    corr_thr, p, itT, σ2_init,
                    out.rsc_psr_u64[end], out.rsc_ber_u64[end],
                    out.rsc_psr_b128[end], out.rsc_ber_b128[end])
        end
    end

    CSV.write(outcsv, out)
    println("\nSaved → $outcsv")
end

main()
