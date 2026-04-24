#!/usr/bin/env julia
# scripts/sweep_jsdc_vs_turboeq.jl
#
# Sweep JSDC INNER+OUTER hyperparameters (including OUTER temperature beta_out)
# and compare against TurboEQ baseline on the SAME selected blocks.
#
# Requires:
#   - lib/compare_3ways.jl (with decode_jsdc_spa exposing inner+outer hypers + beta_out)
#   - lib/TurboEQ.jl
#
# Run:
#   julia --project=. scripts/sweep_jsdc_vs_turboeq.jl --help

using Random, Statistics, Printf
using DataFrames, CSV

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
ensure_linksim_loaded!()

include(joinpath(ROOT, "lib", "compare_3ways.jl"))

if !isdefined(Main, :TurboEQ)
    include(joinpath(ROOT, "lib", "TurboEQ.jl"))
end
using .TurboEQ

# ----------------------------
# Helpers
# ----------------------------
psr64_from_u256(u_hat::Vector{Int}, u_true::Vector{Int}) = sum([
    all(u_hat[64k+1:64k+64] .== u_true[64k+1:64k+64]) for k in 0:3
]) / 4.0

ber_u256(u_hat::Vector{Int}, u_true::Vector{Int}) = mean(u_hat .!= u_true)

function u256hat(x)
    if x isa NamedTuple
        hasproperty(x, :u256_hat) || error("NamedTuple missing :u256_hat keys=$(propertynames(x))")
        return Vector{Int}(Int.(getproperty(x, :u256_hat)))
    elseif x isa AbstractDict
        return Vector{Int}(Int.(x["u256_hat"]))
    elseif x isa AbstractVector
        return Vector{Int}(Int.(x))
    else
        error("Unexpected decoder output type: $(typeof(x))")
    end
end

parse_float_list(s::String) = (ss=replace(strip(s)," "=>""); isempty(ss) ? Float64[] : Float64.(parse.(Float64, split(ss,","))))
parse_int_list(s::String)   = (ss=replace(strip(s)," "=>""); isempty(ss) ? Int[] : Int.(parse.(Int, split(ss,","))))

function beats_turbo(js_psr::Float64, tb_psr::Float64, js_ber::Float64, tb_ber::Float64;
                     eps_psr::Float64=1e-12, eps_ber::Float64=1e-12)
    (js_psr > tb_psr + eps_psr) || (abs(js_psr - tb_psr) <= eps_psr && js_ber < tb_ber - eps_ber)
end

# ----------------------------
# Main
# ----------------------------
function main()
    dataset_ldpc = joinpath(DATA_DIR, "replayswap_qpsk_concat_256_512_1024_from_realdata_donorLS_h20_rho1e-2.jld2")
    dataset_rsc  = joinpath(DATA_DIR, "replayswap_qpsk_RSCconcat_256_512_1024_from_realdata_donorLS_h20_rho1e-2.jld2")
    outdir = joinpath(DATA_DIR, "runs")
    mkpath(outdir)

    # Block selection
    corr_thr = 0.00
    use_nblk = 225
    seed_sel = 12648430
    p = 0.50

    # TurboEQ baseline
    niters = 4
    damp   = 0.1
    M_eq   = 11
    σ2_init = 0.5

    # Pilot boost (multiplies λ_pil_in and λ_pil_out)
    jsdc_pil_boost = 40.0

    # ----------------------------
    # INNER sweep lists (defaults)
    # ----------------------------
    lampar_in_list = [1.0]
    lampil_in_list = [2.0]          # BEFORE boost
    etaz_in_list   = [1e-2]     # focus around your sweet spot
    gamz_in_list   = [1e-1]
    maxit_in_list  = [300]

    # ----------------------------
    # OUTER sweep lists (defaults)
    # ----------------------------
    alpha_out_list  = [2.0]
    beta_out_list   = [0.75, 1.0, 1.25, 1.5, 2.0]   # NEW: temperature
    lampar_out_list = [1.0]
    lampil_out_list = [1.0]                    # BEFORE boost
    etaz_out_list   = [1e-2]
    gamz_out_list   = [2e-2]
    maxit_out_list  = [150]

    exit_on_win = false
    save_best   = true

    # CLI
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a=="--dataset_ldpc"; i+=1; dataset_ldpc = ARGS[i]
        elseif a=="--dataset_rsc"; i+=1; dataset_rsc = ARGS[i]
        elseif a=="--outdir"; i+=1; outdir = ARGS[i]
        elseif a=="--corr"; i+=1; corr_thr=parse(Float64,ARGS[i])
        elseif a=="--nblk"; i+=1; use_nblk=parse(Int,ARGS[i])
        elseif a=="--seed_sel"; i+=1; seed_sel=parse(Int,ARGS[i])
        elseif a=="--p"; i+=1; p=parse(Float64,ARGS[i])

        elseif a=="--niters"; i+=1; niters=parse(Int,ARGS[i])
        elseif a=="--damp"; i+=1; damp=parse(Float64,ARGS[i])
        elseif a=="--Meq"; i+=1; M_eq=parse(Int,ARGS[i])
        elseif a=="--sigma2_init"; i+=1; σ2_init=parse(Float64,ARGS[i])

        elseif a=="--jsdc_pil_boost"; i+=1; jsdc_pil_boost=parse(Float64,ARGS[i])

        # INNER lists
        elseif a=="--lampar_in_list"; i+=1; lampar_in_list=parse_float_list(ARGS[i])
        elseif a=="--lampil_in_list"; i+=1; lampil_in_list=parse_float_list(ARGS[i])
        elseif a=="--etaz_in_list";   i+=1; etaz_in_list=parse_float_list(ARGS[i])
        elseif a=="--gamz_in_list";   i+=1; gamz_in_list=parse_float_list(ARGS[i])
        elseif a=="--maxit_in_list";  i+=1; maxit_in_list=parse_int_list(ARGS[i])

        # OUTER lists
        elseif a=="--alpha_out_list"; i+=1; alpha_out_list=parse_float_list(ARGS[i])
        elseif a=="--beta_out_list";  i+=1; beta_out_list=parse_float_list(ARGS[i])     # NEW
        elseif a=="--lampar_out_list"; i+=1; lampar_out_list=parse_float_list(ARGS[i])
        elseif a=="--lampil_out_list"; i+=1; lampil_out_list=parse_float_list(ARGS[i])
        elseif a=="--etaz_out_list";   i+=1; etaz_out_list=parse_float_list(ARGS[i])
        elseif a=="--gamz_out_list";   i+=1; gamz_out_list=parse_float_list(ARGS[i])
        elseif a=="--maxit_out_list";  i+=1; maxit_out_list=parse_int_list(ARGS[i])

        elseif a=="--exit_on_win"; i+=1; exit_on_win = (parse(Int,ARGS[i]) != 0)
        elseif a=="--save_best";   i+=1; save_best = (parse(Int,ARGS[i]) != 0)

        elseif a=="--help" || a=="-h"
            println("""
Usage:
  julia --project=. scripts/sweep_jsdc_vs_turboeq.jl [args]

Core:
  --p 0.20 --corr 0.40 --nblk 225 --seed_sel 12648430

TurboEQ:
  --niters 6 --damp 0.6 --Meq 11 --sigma2_init 0.5

JSDC:
  --jsdc_pil_boost 40

Inner sweep lists:
  --lampar_in_list "1.0"
  --lampil_in_list "2.0,3.0,4.0"
  --etaz_in_list   "0.006,0.01,0.014"
  --gamz_in_list   "0.06,0.1,0.15"
  --maxit_in_list  "300"

Outer sweep lists:
  --alpha_out_list "2.0,2.25,2.5"
  --beta_out_list  "0.75,1.0,1.25,1.5,2.0"   (temperature)
  --lampar_out_list "1.0"
  --lampil_out_list "1.0,2.0"
  --etaz_out_list   "0.005,0.01,0.02"
  --gamz_out_list   "0.02,0.04"
  --maxit_out_list  "150"

Control:
  --exit_on_win 0/1
  --save_best 0/1
""")
            return
        else
            error("Unknown arg: $a")
        end
        i += 1
    end
    mkpath(outdir)

    # Load datasets
    dl0 = load_dataset_any(dataset_ldpc)
    dr0 = load_dataset_any(dataset_rsc)

    nblk_all = size(dl0["y_qpsk_swapped"], 1)
    @assert size(dr0["y_qpsk_swapped"], 1) == nblk_all

    eligible = findall((dl0["corr_donor"] .>= corr_thr) .& (dr0["corr_donor"] .>= corr_thr))
    rng = MersenneTwister(seed_sel)
    shuffle!(rng, eligible)
    blk_list = eligible[1:min(use_nblk, length(eligible))]
    isempty(blk_list) && error("No eligible blocks at corr_thr=$corr_thr")

    itlv_l = get_interleaver(dl0["meta_out"])
    itlv_r = get_interleaver(dr0["meta_out"])

    # init codes
    codeO, colsO, idrowsO, _ = initcode(k1, n1, npc_local)
    codeI, colsI, idrowsI, _ = initcode(k2, n2, npc_local)
    codeO.icols === nothing && (encode(codeO, zeros(Int, k1)); nothing)
    codeI.icols === nothing && (encode(codeI, zeros(Int, k2)); nothing)

    HO = get_H_sparse(codeO)
    HI = get_H_sparse(codeI)

    # For jsdc_qpsk_manual, idrows are already parity neighbor lists
    parityO = idrowsO
    parityI = idrowsI

    println("==============================================================")
    @printf("SWEEP JSDC (inner+outer + beta_out) vs TurboEQ | p=%.2f corr=%.2f using=%d\n", p, corr_thr, length(blk_list))
    @printf("TurboEQ: niters=%d damp=%.2f Meq=%d\n", niters, damp, M_eq)
    @printf("Pilot boost: %.1f\n", jsdc_pil_boost)
    println("--------------------------------------------------------------")
    @printf("Inner grid: lampar=%s lampil=%s etaz=%s gamz=%s maxit=%s\n",
            string(lampar_in_list), string(lampil_in_list), string(etaz_in_list), string(gamz_in_list), string(maxit_in_list))
    @printf("Outer grid: alpha=%s beta=%s lampar=%s lampil=%s etaz=%s gamz=%s maxit=%s\n",
            string(alpha_out_list), string(beta_out_list), string(lampar_out_list), string(lampil_out_list),
            string(etaz_out_list), string(gamz_out_list), string(maxit_out_list))
    println("==============================================================")

    # ------------------------------------------------------------
    # Precompute TurboEQ baseline once
    # ------------------------------------------------------------
    tb_psr = Vector{Float64}(undef, length(blk_list))
    tb_ber = Vector{Float64}(undef, length(blk_list))

    for (ii,b) in enumerate(blk_list)
        yr = ComplexF64.(vec(dr0["y_qpsk_swapped"][b, :]))
        ur = Int.(vec(dr0["u256_mat"][b, :]))
        b512r  = Int.(vec(dr0["b512_mat"][b, :]))
        b512ir = haskey(dr0, "b512_i_mat") ? Int.(vec(dr0["b512_i_mat"][b, :])) : b512r
        b1024r = Int.(vec(dr0["b1024_mat"][b, :]))
        hr = ComplexF64.(vec(dr0["h_blk_mat"][b, :]))

        tb = TurboEQ.decode_turboeq(yr, ur, b512r, b512ir, b1024r, hr, itlv_r;
                                    p=p, niters=niters, damp=damp, M_eq=M_eq, σ2_init=σ2_init)
        utb = u256hat(tb)
        tb_psr[ii] = psr64_from_u256(utb, ur)
        tb_ber[ii] = ber_u256(utb, ur)

        if ii == 1 || ii % 50 == 0 || ii == length(blk_list)
            @printf("  Turbo baseline %d/%d | meanPSR=%.3f meanBER=%.3f\n",
                    ii, length(blk_list), mean(tb_psr[1:ii]), mean(tb_ber[1:ii]))
        end
    end

    tb_psr_mean = mean(tb_psr)
    tb_ber_mean = mean(tb_ber)
    @printf("\nTurboEQ baseline: meanPSR64=%.4f meanBER=%.6f\n\n", tb_psr_mean, tb_ber_mean)

    # ------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------
    best = (; psr=-Inf, ber=Inf,
            lampar_in=NaN, lampil_in=NaN, etaz_in=NaN, gamz_in=NaN, maxit_in=0,
            alpha_out=NaN, beta_out=NaN, lampar_out=NaN, lampil_out=NaN, etaz_out=NaN, gamz_out=NaN, maxit_out=0)
    best_df = nothing

    cfg_id = 0
    total_cfg =
        length(lampar_in_list)*length(lampil_in_list)*length(etaz_in_list)*length(gamz_in_list)*length(maxit_in_list) *
        length(alpha_out_list)*length(beta_out_list)*length(lampar_out_list)*length(lampil_out_list)*length(etaz_out_list)*length(gamz_out_list)*length(maxit_out_list)

    for lampar_in in lampar_in_list, lampil_in in lampil_in_list, etaz_in in etaz_in_list, gamz_in in gamz_in_list, maxit_in in maxit_in_list,
        aout in alpha_out_list, bout in beta_out_list, lpar in lampar_out_list, lpil in lampil_out_list, etaz in etaz_out_list, gamz in gamz_out_list, maxitO in maxit_out_list

        cfg_id += 1

        js_psr_vec = Vector{Float64}(undef, length(blk_list))
        js_ber_vec = Vector{Float64}(undef, length(blk_list))

        for (ii,b) in enumerate(blk_list)
            yl = ComplexF64.(vec(dl0["y_qpsk_swapped"][b, :]))
            ul = Int.(vec(dl0["u256_mat"][b, :]))
            b512l  = Int.(vec(dl0["b512_mat"][b, :]))
            b1024l = Int.(vec(dl0["b1024_mat"][b, :]))
            hl = ComplexF64.(vec(dl0["h_blk_mat"][b, :]))

            js = decode_jsdc_spa(yl, ul, b512l, b1024l, hl,
                                 codeO, idrowsO, HO, colsO, parityO,
                                 codeI, idrowsI, HI, colsI, parityI,
                                 itlv_l;
                                 p=p,
                                 jsdc_pil_boost=jsdc_pil_boost,
                                 # inner
                                 λ_par_in=lampar_in,
                                 λ_pil_in=lampil_in,
                                 η_z_in=etaz_in,
                                 γ_z_in=gamz_in,
                                 maxit_in=maxit_in,
                                 # outer
                                 alpha_out=aout,
                                 beta_out=bout,
                                 λ_par_out=lpar,
                                 λ_pil_out=lpil,
                                 η_z_out=etaz,
                                 γ_z_out=gamz,
                                 maxit_out=maxitO)

            ujs = u256hat(js)
            js_psr_vec[ii] = psr64_from_u256(ujs, ul)
            js_ber_vec[ii] = ber_u256(ujs, ul)
        end

        js_psr_mean = mean(js_psr_vec)
        js_ber_mean = mean(js_ber_vec)

        @printf("cfg %4d/%d | IN(lp=%.2f pil=%.2f ez=%.2e gz=%.2e it=%d) OUT(a=%.2f β=%.2f lp=%.2f pil=%.2f ez=%.2e gz=%.2e it=%d) | J:PSR=%.3f BER=%.3f vs T:PSR=%.3f BER=%.3f\n",
                cfg_id, total_cfg,
                lampar_in, lampil_in, etaz_in, gamz_in, maxit_in,
                aout, bout, lpar, lpil, etaz, gamz, maxitO,
                js_psr_mean, js_ber_mean, tb_psr_mean, tb_ber_mean)

        # best update
        if beats_turbo(js_psr_mean, tb_psr_mean, js_ber_mean, tb_ber_mean) ||
           (js_psr_mean > best.psr + 1e-12) ||
           (abs(js_psr_mean - best.psr) <= 1e-12 && js_ber_mean < best.ber)

            best = (; psr=js_psr_mean, ber=js_ber_mean,
                    lampar_in=lampar_in, lampil_in=lampil_in, etaz_in=etaz_in, gamz_in=gamz_in, maxit_in=maxit_in,
                    alpha_out=aout, beta_out=bout, lampar_out=lpar, lampil_out=lpil, etaz_out=etaz, gamz_out=gamz, maxit_out=maxitO)

            best_df = DataFrame(
                blk = Int.(blk_list),
                corr = Float64.(dl0["corr_donor"][blk_list]),
                tb_psr64 = tb_psr,
                tb_ber   = tb_ber,
                js_psr64 = js_psr_vec,
                js_ber   = js_ber_vec,
            )
        end

        if beats_turbo(js_psr_mean, tb_psr_mean, js_ber_mean, tb_ber_mean)
            println(">>> FOUND WINNER vs TurboEQ.")
            @printf(">>> BEST J:PSR=%.4f BER=%.6f | IN(lp=%.2f pil=%.2f ez=%.2e gz=%.2e it=%d) OUT(a=%.2f β=%.2f lp=%.2f pil=%.2f ez=%.2e gz=%.2e it=%d)\n",
                    best.psr, best.ber,
                    best.lampar_in, best.lampil_in, best.etaz_in, best.gamz_in, best.maxit_in,
                    best.alpha_out, best.beta_out, best.lampar_out, best.lampil_out, best.etaz_out, best.gamz_out, best.maxit_out)

            if save_best && best_df !== nothing
                tag = @sprintf("p%.2f__corr%.2f__n%d__WIN", p, corr_thr, length(blk_list))
                out_csv = joinpath(outdir, "jsdc_sweep_vs_turboeq__" * tag * ".csv")
                CSV.write(out_csv, best_df)
                println("Saved winner CSV → $out_csv")
            end
            exit_on_win && return
        end
    end

    println("\nNo config in this grid beat TurboEQ.")
    @printf("Best seen:\n")
    @printf("  IN : lpar=%.2f lpil=%.2f etaz=%.2e gamz=%.2e it=%d\n", best.lampar_in, best.lampil_in, best.etaz_in, best.gamz_in, best.maxit_in)
    @printf("  OUT: a=%.2f beta=%.2f lpar=%.2f lpil=%.2f etaz=%.2e gamz=%.2e it=%d\n",
            best.alpha_out, best.beta_out, best.lampar_out, best.lampil_out, best.etaz_out, best.gamz_out, best.maxit_out)
    @printf("  J:PSR=%.4f BER=%.6f (Turbo PSR=%.4f BER=%.6f)\n",
            best.psr, best.ber, tb_psr_mean, tb_ber_mean)

    if save_best && best_df !== nothing
        tag = @sprintf("p%.2f__corr%.2f__n%d__BEST", p, corr_thr, length(blk_list))
        out_csv = joinpath(outdir, "jsdc_sweep_vs_turboeq__" * tag * ".csv")
        CSV.write(out_csv, best_df)
        println("Saved best CSV → $out_csv")
    end
end

main()
