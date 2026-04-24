#!/usr/bin/env julia
# scripts/debug_turboeq_bpsk_rsc64_128.jl
#
# Sanity/debug runner for BPSK RSC(64->128) TurboEQ-style decode.
#
# IMPORTANT FIX:
#   - Reports BOTH:
#       * b128(ch): hard decision on channel LLRs (pre-BCJR)
#       * b128(post): hard decision on BCJR codeword posteriors (post-BCJR)
#     This avoids the confusing situation where u64 is perfect but b128 looks bad
#     simply because you were grading the pre-decoder LLRs.
#
# Example:
#   julia --project=. scripts/debug_turboeq_bpsk_rsc64_128.jl
#
#   julia --project=. scripts/debug_turboeq_bpsk_rsc64_128.jl \
#     --ps "0,0.1,0.2,0.3" --corr 0.10 --nblk 225 --iters 4 --sigma2 0.30
#
using Random, Statistics, Printf
using JLD2, DataFrames

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
ensure_linksim_loaded!()  # harmless; keeps your environment consistent

include(joinpath(ROOT, "lib", "TurboEQ_BPSK_RSC64_128.jl"))
using .TurboEQ_BPSK_RSC64_128

# ----------------------------
# CLI helpers
# ----------------------------
function parse_floats(s::String)
    t = replace(strip(s), " " => "")
    isempty(t) && return Float64[]
    return Float64.(parse.(Float64, split(t, ",")))
end

@inline ber(a::AbstractVector{Int}, b::AbstractVector{Int}) = mean(a .!= b)
@inline psr_pkt(a::AbstractVector{Int}, b::AbstractVector{Int}) = all(a .== b) ? 1.0 : 0.0

# ----------------------------
# Main
# ----------------------------
function main()
    rsc_path  = joinpath(DATA_DIR, "replayswap_bpsk_RSC_64_128_from_realdata_donorLS_h20_rho1e-2.jld2")

    corr_thr  = 0.10
    use_nblk  = 225
    seed_sel  = 12648430
    start     = 1

    ps        = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    iters     = 3
    σ2_init   = 0.30
    eq_σ2_it  = 1
    llr_clip  = 25.0

    # simple ARGS parse
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a=="--rsc"; i+=1; rsc_path = ARGS[i]
        elseif a=="--corr"; i+=1; corr_thr = parse(Float64, ARGS[i])
        elseif a=="--nblk"; i+=1; use_nblk = parse(Int, ARGS[i])
        elseif a=="--seed_sel"; i+=1; seed_sel = parse(Int, ARGS[i])
        elseif a=="--start"; i+=1; start = parse(Int, ARGS[i])
        elseif a=="--ps"; i+=1; ps = parse_floats(ARGS[i])
        elseif a=="--iters"; i+=1; iters = parse(Int, ARGS[i])
        elseif a=="--sigma2"; i+=1; σ2_init = parse(Float64, ARGS[i])
        elseif a=="--eq_sigma2_iters"; i+=1; eq_σ2_it = parse(Int, ARGS[i])
        elseif a=="--llr_clip"; i+=1; llr_clip = parse(Float64, ARGS[i])
        elseif a=="--help" || a=="-h"
            println("""
Usage:
  julia --project=. scripts/debug_turboeq_bpsk_rsc64_128.jl [args]

Args:
  --rsc   <path>   dataset .jld2
  --corr  <float>  corr_thr (default 0.10)
  --nblk  <int>    number of blocks (default 225)
  --seed_sel <int> selection seed
  --start <int>    start index in eligible list (1-based)

Decode:
  --ps    "0,0.1,0.2" pilot fractions
  --iters <int>       turbo iters (default 3)
  --sigma2 <float>    σ2_init (default 0.30)
  --eq_sigma2_iters <int> inner σ² refine inside equalizer (default 1)
  --llr_clip <float>  LLR clip (default 25)

Notes:
  - Reports b128(ch) metrics from llr128_ch (pre-BCJR)
  - Reports b128(post) metrics from llr128_post (post-BCJR)
""")
            return
        else
            error("Unknown arg: $a")
        end
        i += 1
    end

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
    @printf("BPSK RSC(64->128) TurboEQ-style sanity | corr_thr=%.2f | using=%d/%d blocks\n",
            corr_thr, length(blk_list), nblk_all)
    @printf("ps = %s | iters=%d | σ2_init=%.3f | eq_σ2_iters=%d | llr_clip=%.1f\n",
            string(ps), iters, σ2_init, eq_σ2_it, llr_clip)
    println("==============================================================")

    for p in ps
        psr_u = Float64[]
        ber_u = Float64[]

        psr_b_post = Float64[]
        ber_b_post = Float64[]

        ber_b_ch = Float64[]   # channel-only b128 BER (pre-BCJR)

        pre_b = Float64[]      # same as ber_b_ch (kept for backwards familiarity)
        sig2s = Float64[]

        for (ii, b) in enumerate(blk_list)
            y = ComplexF64.(vec(ymat[b, :]))
            u_true = Int.(vec(umat[b, :]))
            b_true = Int.(vec(bmat[b, :]))
            h = ComplexF64.(vec(hmat[b, :]))

            out = decode_turboeq_rsc_bpsk(
                y, h, u_true, b_true;
                p=p, turbo_iters=iters, σ2_init=σ2_init,
                eq_σ2_iters=eq_σ2_it, llr_clip=llr_clip
            )

            u_hat = out.u64_hat
            b_hat_ch   = hard_from_llr(out.llr128_ch)
            b_hat_post = hard_from_llr(out.llr128_post)

            push!(psr_u, psr_pkt(u_hat, u_true))
            push!(ber_u, ber(u_hat, u_true))

            # post-BCJR codeword metrics (the meaningful "decoded b128")
            push!(psr_b_post, psr_pkt(b_hat_post, b_true))
            push!(ber_b_post, ber(b_hat_post, b_true))

            # channel-only b128 BER (pre-BCJR)
            chber = ber(b_hat_ch, b_true)
            push!(ber_b_ch, chber)
            push!(pre_b, chber)

            push!(sig2s, out.sigma2_hat)

            if ii == 1 || ii % 50 == 0 || ii == length(blk_list)
                @printf("  p=%.2f blk %d/%d | u64 PSR=%.0f BER=%.3f | b128(post) BER=%.3f | b128(ch) BER=%.3f\n",
                        p, ii, length(blk_list),
                        psr_u[end], ber_u[end],
                        ber_b_post[end], ber_b_ch[end])
            end
        end

        @printf("\nSUMMARY p=%.2f | u64 PSR=%.3f BER=%.3f | b128(post) PSR=%.3f BER=%.3f | b128(ch) BER=%.3f | σ2(p50)=%.4f\n\n",
                p,
                mean(psr_u), mean(ber_u),
                mean(psr_b_post), mean(ber_b_post),
                mean(ber_b_ch),
                median(sig2s))
    end

    println("Done.")
end

main()
