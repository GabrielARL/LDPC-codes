#!/usr/bin/env julia
# scripts/psr_bpsk_rsc_psweep_only.jl
#
# RSC-only runner:
#   - Replay-swap BPSK RSC(64->128)
#   - TurboEQ (decode_turboeq_rsc_bpsk)
#   Sweep: pilot ratio p
#
# Output (default):
#   - data/runs/psr_bpsk_rsc_turbo.csv
#
# Examples:
#   julia --project=. scripts/psr_bpsk_rsc_psweep_only.jl
#   julia --project=. scripts/psr_bpsk_rsc_psweep_only.jl --ps "0,0.1,0.2,0.3,0.4,0.5" --nblk 200

using Random, Printf, Statistics
using JLD2, DataFrames, CSV

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
const LS = ensure_linksim_loaded!()  # keep this: TurboEQ code expects LinkSim loaded

# TurboEQ decoder for RSC replay-swap BPSK RSC64->128
include(joinpath(ROOT, "lib", "TurboEQ_BPSK_RSC64_128.jl"))
using .TurboEQ_BPSK_RSC64_128: decode_turboeq_rsc_bpsk

# ----------------------------
# Small utils
# ----------------------------
@inline hard_from_llr(L::AbstractVector{<:Real}) = Int.(L .< 0)
@inline ber(a::AbstractVector{Int}, b::AbstractVector{Int}) = mean(a .!= b)
@inline psr_pkt(a::AbstractVector{Int}, b::AbstractVector{Int}) = all(a .== b) ? 1.0 : 0.0

parse_int(s::String) = parse(Int, strip(s))
function parse_floats(s::String)
    t = replace(strip(s), " " => "")
    isempty(t) && return Float64[]
    return Float64.(parse.(Float64, split(t, ",")))
end

# ----------------------------
# RSC TurboEQ sweep runner
# ----------------------------
function run_rsc_turbo_psweep(; rsc_path::String,
                              corr_thr::Float64,
                              nblk::Int,
                              seed_sel::Int,
                              start::Int,
                              ps::Vector{Float64},
                              turbo_iters::Int,
                              σ2_init::Float64,
                              eq_σ2_iters::Int,
                              llr_clip::Float64,
                              out_csv::String)

    isfile(rsc_path) || error("Missing RSC dataset: $rsc_path")
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

    df = DataFrame(
        p=Float64[], blk=Int[], corr=Float64[],
        u64_psr=Float64[], u64_ber=Float64[],
        b128_post_psr=Float64[], b128_post_ber=Float64[],
        b128_ch_ber=Float64[],
        sigma2_final=Float64[]
    )

    println("==============================================================")
    @printf("RSC TurboEQ PSWEEP | blocks=%d/%d corr_thr=%.2f ps=%s\n",
            length(blk_list), length(eligible), corr_thr, string(ps))
    @printf("TurboEQ: iters=%d | σ2_init=%.3f | eq_σ2_iters=%d | llr_clip=%.1f\n",
            turbo_iters, σ2_init, eq_σ2_iters, llr_clip)
    println("==============================================================")

    for p in ps
        @printf("\n--- RSC p=%.3f ---\n", p)
        for (ii, b) in enumerate(blk_list)
            y = ComplexF64.(vec(ymat[b, :]))
            u_true = Int.(vec(umat[b, :]))
            b_true = Int.(vec(bmat[b, :]))
            hfull = ComplexF64.(vec(hmat[b, :]))
            Lh = min(length(hfull), length(y))
            h_use = hfull[1:Lh]

            tout = decode_turboeq_rsc_bpsk(
                y, h_use, u_true, b_true;
                p=p,
                turbo_iters=turbo_iters,
                σ2_init=σ2_init,
                eq_σ2_iters=eq_σ2_iters,
                llr_clip=llr_clip
            )

            u_hat = Vector{Int}(tout.u64_hat)
            b_hat_post = hard_from_llr(tout.llr128_post)
            b_hat_ch   = hard_from_llr(tout.llr128_ch)

            push!(df, (
                p=p, blk=b, corr=Float64(corr[b]),
                u64_psr=psr_pkt(u_hat, u_true),
                u64_ber=ber(u_hat, u_true),
                b128_post_psr=psr_pkt(b_hat_post, b_true),
                b128_post_ber=ber(b_hat_post, b_true),
                b128_ch_ber=ber(b_hat_ch, b_true),
                sigma2_final = hasproperty(tout, :sigma2_final) ? Float64(getproperty(tout, :sigma2_final)) : NaN
            ))

            if ii == 1 || ii % 50 == 0 || ii == length(blk_list)
                @printf("  blk %d/%d | u64 PSR=%.3f b128(post) PSR=%.3f\n",
                        ii, length(blk_list),
                        psr_pkt(u_hat, u_true),
                        psr_pkt(b_hat_post, b_true))
            end
        end
    end

    mkpath(dirname(out_csv))
    CSV.write(out_csv, df)
    println("\nSaved RSC TurboEQ sweep → $out_csv")
    return df
end

# ----------------------------
# MAIN
# ----------------------------
function main()
    # --- defaults ---
    rsc_path = joinpath(DATA_DIR, "replayswap_bpsk_RSC_64_128_from_realdata_donorLS_h20_rho1e-2.jld2")
    corr_thr = 0.10
    nblk = 200
    seed_sel = 12648430
    start = 1
    ps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    turbo_iters = 2
    σ2_init = 1.30
    eq_σ2_iters = 1
    llr_clip = 25.0

    out_csv = joinpath(DATA_DIR, "runs", "psr_bpsk_rsc_turbo.csv")

    # --- CLI ---
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--rsc_path"; i+=1; rsc_path = ARGS[i]
        elseif a == "--corr" || a == "--corr_rsc"; i+=1; corr_thr = parse(Float64, ARGS[i])
        elseif a == "--nblk"; i+=1; nblk = parse_int(ARGS[i])
        elseif a == "--seed_sel"; i+=1; seed_sel = parse_int(ARGS[i])
        elseif a == "--start"; i+=1; start = parse_int(ARGS[i])
        elseif a == "--ps"; i+=1; ps = parse_floats(ARGS[i])

        elseif a == "--turbo_iters"; i+=1; turbo_iters = parse_int(ARGS[i])
        elseif a == "--σ2_init" || a == "--sigma2_init"; i+=1; σ2_init = parse(Float64, ARGS[i])
        elseif a == "--eq_σ2_iters" || a == "--eq_sigma2_iters"; i+=1; eq_σ2_iters = parse_int(ARGS[i])
        elseif a == "--llr_clip"; i+=1; llr_clip = parse(Float64, ARGS[i])
        elseif a == "--out_csv"; i+=1; out_csv = ARGS[i]

        elseif a == "--help" || a == "-h"
            println("""
Usage:
  julia --project=. scripts/psr_bpsk_rsc_psweep_only.jl [args]

RSC dataset:
  --rsc_path <path>
  --corr <float>            (corr threshold, default 0.10)
  --nblk <int>              (default 200)
  --seed_sel <int>          (default 12648430)
  --start <int>             (start index within eligible blocks, default 1)
  --ps "0,0.1,0.2,0.3,0.4,0.5"

TurboEQ knobs:
  --turbo_iters <int>       (default 2)
  --σ2_init <float>         (default 1.30)
  --eq_σ2_iters <int>       (default 1)
  --llr_clip <float>        (default 25.0)

Output:
  --out_csv <path>          (default data/runs/psr_bpsk_rsc_turbo.csv)
""")
            return
        else
            error("Unknown arg: $a")
        end
        i += 1
    end

    println("==============================================================")
    @printf("RSC-only PSWEEP | corr_thr=%.2f nblk=%d ps=%s\n", corr_thr, nblk, string(ps))
    println("==============================================================")

    run_rsc_turbo_psweep(; rsc_path=rsc_path, corr_thr=corr_thr,
                         nblk=nblk, seed_sel=seed_sel, start=start,
                         ps=ps,
                         turbo_iters=turbo_iters, σ2_init=σ2_init, eq_σ2_iters=eq_σ2_iters,
                         llr_clip=llr_clip,
                         out_csv=out_csv)

    println("\nDone.")
end

main()
