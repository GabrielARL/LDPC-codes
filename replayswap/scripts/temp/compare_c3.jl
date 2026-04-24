#!/usr/bin/env julia
using Random, Statistics, Printf
using SparseArrays
using DataFrames, CSV

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
ensure_linksim_loaded!()

include(joinpath(ROOT, "lib", "compare_3ways.jl"))

if !isdefined(Main, :TurboEQ)
    include(joinpath(ROOT, "lib", "TurboEQ.jl"))
end
using .TurboEQ

# NEW: JSDC-TURBO rounds
include(joinpath(ROOT, "lib", "jsdc_turbo_rounds.jl"))

psr64_from_u256(u_hat::Vector{Int}, u_true::Vector{Int}) = sum([
    all(u_hat[64k+1:64k+64] .== u_true[64k+1:64k+64]) for k in 0:3
]) / 4.0
ber_u256(u_hat::Vector{Int}, u_true::Vector{Int}) = mean(u_hat .!= u_true)

function u256hat(x)
    if x isa NamedTuple
        return Vector{Int}(Int.(getproperty(x, :u256_hat)))
    elseif x isa AbstractDict
        return Vector{Int}(Int.(x["u256_hat"]))
    else
        return Vector{Int}(Int.(x))
    end
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

function main()
    dataset_ldpc = joinpath(DATA_DIR, "replayswap_qpsk_concat_256_512_1024_from_realdata_donorLS_h20_rho1e-2.jld2")
    dataset_rsc  = joinpath(DATA_DIR, "replayswap_qpsk_RSCconcat_256_512_1024_from_realdata_donorLS_h20_rho1e-2.jld2")
    outcsv = joinpath(DATA_DIR, "runs", "compare_c3_lean.csv")

    corr_thr = 0.10
    use_nblk = 225
    seed_sel = 12648430
    p = 0.10

    # TurboEQ params
    niters = 6
    damp   = 0.1
    M_eq   = 6
    σ2_init = 1.5

    # JSDC baseline (single config)
    inner_pil_boost = 40.0
    alpha_out = 1.5
    lampar_out = 1.0
    lampil_out = 0.5
    etaz_out = 1e-2
    gamz_out = 1.0e-3
    maxit_out = 300

    # NEW: JSDC-TURBO rounds knobs
    jt_enable = true
    jt_rounds = 2
    jt_prior_w_in  = 0.6
    jt_prior_w_out = 1.0
    jt_prior_damp  = 0.3
    jt_prior_clip  = 8.0
    jt_outer_alpha = 1.5

    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a=="--dataset_ldpc"; i+=1; dataset_ldpc=ARGS[i]
        elseif a=="--dataset_rsc"; i+=1; dataset_rsc=ARGS[i]
        elseif a=="--outcsv"; i+=1; outcsv=ARGS[i]
        elseif a=="--corr"; i+=1; corr_thr=parse(Float64,ARGS[i])
        elseif a=="--nblk"; i+=1; use_nblk=parse(Int,ARGS[i])
        elseif a=="--seed_sel"; i+=1; seed_sel=parse(Int,ARGS[i])
        elseif a=="--p"; i+=1; p=parse(Float64,ARGS[i])
        elseif a=="--jt_enable"; i+=1; jt_enable = (parse(Int,ARGS[i]) != 0)
        elseif a=="--jt_rounds"; i+=1; jt_rounds = parse(Int,ARGS[i])
        else
            error("Unknown arg: $a")
        end
        i += 1
    end

    mkpath(dirname(outcsv))

    dl0 = load_dataset_any(dataset_ldpc)
    dr0 = load_dataset_any(dataset_rsc)

    nblk = size(dl0["y_qpsk_swapped"], 1)
    @assert size(dr0["y_qpsk_swapped"], 1) == nblk

    eligible = findall((dl0["corr_donor"] .>= corr_thr) .& (dr0["corr_donor"] .>= corr_thr))
    rng = MersenneTwister(seed_sel)
    shuffle!(rng, eligible)
    blk_list = eligible[1:min(use_nblk, length(eligible))]
    isempty(blk_list) && error("No eligible blocks at corr_thr=$corr_thr")

    itlv_l = get_interleaver(dl0["meta_out"])
    itlv_r = get_interleaver(dr0["meta_out"])

    codeO, colsO, idrowsO, _ = initcode(k1, n1, npc_local)
    codeI, colsI, idrowsI, _ = initcode(k2, n2, npc_local)
    codeO.icols === nothing && (encode(codeO, zeros(Int, k1)); nothing)
    codeI.icols === nothing && (encode(codeI, zeros(Int, k2)); nothing)

    HO = get_H_sparse(codeO)
    HI = get_H_sparse(codeI)
    parityO = build_parity_indices(HO)
    parityI = build_parity_indices(HI)

    df = DataFrame(
        blk=Int[], corr=Float64[],
        turbo_psr64=Float64[], turbo_ber=Float64[],
        eq_psr64=Float64[],    eq_ber=Float64[],
        jsdc_psr64=Float64[],  jsdc_ber=Float64[],
        jsdcT_psr64=Float64[], jsdcT_ber=Float64[],
    )

    println("==============================================================")
    @printf("COMPARE: p=%.2f corr=%.2f using=%d | JSDC-TURBO=%s rounds=%d\n",
            p, corr_thr, length(blk_list), jt_enable ? "ON" : "OFF", jt_rounds)
    println("==============================================================")

    for (ii,b) in enumerate(blk_list)
        yl = ComplexF64.(vec(dl0["y_qpsk_swapped"][b, :]))
        ul = Int.(vec(dl0["u256_mat"][b, :]))
        b512l  = Int.(vec(dl0["b512_mat"][b, :]))
        b512il = haskey(dl0, "b512_i_mat") ? Int.(vec(dl0["b512_i_mat"][b, :])) : b512l
        b1024l = Int.(vec(dl0["b1024_mat"][b, :]))
        hl = ComplexF64.(vec(dl0["h_blk_mat"][b, :]))

        yr = ComplexF64.(vec(dr0["y_qpsk_swapped"][b, :]))
        ur = Int.(vec(dr0["u256_mat"][b, :]))
        b512r  = Int.(vec(dr0["b512_mat"][b, :]))
        b512ir = haskey(dr0, "b512_i_mat") ? Int.(vec(dr0["b512_i_mat"][b, :])) : b512r
        b1024r = Int.(vec(dr0["b1024_mat"][b, :]))
        hr = ComplexF64.(vec(dr0["h_blk_mat"][b, :]))

        tb  = TurboEQ.decode_turboeq(yr, ur, b512r, b512ir, b1024r, hr, itlv_r;
                                     p=p, niters=niters, damp=damp, M_eq=M_eq, σ2_init=σ2_init)
        utb = u256hat(tb)
        tb_psr = psr64_from_u256(utb, ur)
        tb_ber = ber_u256(utb, ur)

        es = decode_eq_spa(yl, ul, b512l, b1024l, hl,
                           codeO, colsO, idrowsO, HO,
                           codeI, colsI, idrowsI, HI,
                           itlv_l; p=p, M_eq=M_eq)
        ues = u256hat(es)
        es_psr = psr64_from_u256(ues, ul)
        es_ber = ber_u256(ues, ul)

        js = decode_jsdc_spa(yl, ul, b512l, b1024l, hl,
                             codeO, idrowsO, HO, colsO, parityO,
                             codeI, idrowsI, HI, colsI, parityI,
                             itlv_l;
                             p=p, jsdc_pil_boost=inner_pil_boost,
                             alpha_out=alpha_out,
                             λ_par_out=lampar_out,
                             λ_pil_out=lampil_out,
                             η_z_out=etaz_out,
                             γ_z_out=gamz_out,
                             maxit_out=maxit_out)
        ujs = u256hat(js)
        js_psr = psr64_from_u256(ujs, ul)
        js_ber = ber_u256(ujs, ul)

        jt_psr = NaN; jt_ber = NaN
        if jt_enable
            jt = decode_jsdc_turbo_rounds(
                yl, ul, b512l, b512il, b1024l, hl,
                codeO, idrowsO, HO, colsO, parityO,
                codeI, idrowsI, HI, colsI, parityI,
                itlv_l;
                p=p,
                jsdc_pil_boost=inner_pil_boost,
                rounds=jt_rounds,
                prior_w_in=jt_prior_w_in,
                prior_w_out=jt_prior_w_out,
                prior_damp=jt_prior_damp,
                prior_clip=jt_prior_clip,
                outer_alpha=jt_outer_alpha,
            )
            ujt = u256hat(jt)
            jt_psr = psr64_from_u256(ujt, ul)
            jt_ber = ber_u256(ujt, ul)
        end

        c = Float64(min(dl0["corr_donor"][b], dr0["corr_donor"][b]))
        push!(df, (blk=b, corr=c,
                   turbo_psr64=tb_psr, turbo_ber=tb_ber,
                   eq_psr64=es_psr, eq_ber=es_ber,
                   jsdc_psr64=js_psr, jsdc_ber=js_ber,
                   jsdcT_psr64=jt_psr, jsdcT_ber=jt_ber))

        if ii == 1 || ii % 10 == 0 || ii == length(blk_list)
            @printf("  blk %d/%d | PSR64: T=%.3f EQ=%.3f J=%.3f JT=%.3f\n",
                    ii, length(blk_list), tb_psr, es_psr, js_psr, jt_psr)
        end
    end

    CSV.write(outcsv, df)
    println("Saved → $outcsv")

    @printf("Mean PSR64: Turbo=%.3f EQ=%.3f JSDC=%.3f JT=%.3f\n",
            mean(df.turbo_psr64), mean(df.eq_psr64), mean(df.jsdc_psr64), mean(skipmissing(df.jsdcT_psr64)))
end

main()
