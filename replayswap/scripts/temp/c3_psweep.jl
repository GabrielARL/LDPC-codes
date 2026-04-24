#!/usr/bin/env julia
# scripts/c3_psweep.jl
#
# Compare-3ways pilot sweep (DFEC):
#   - Sweeps pilot ratio p (default: 0.0:0.1:0.5)
#   - Uses SAME selected blocks for all p (selected once by corr_thr + seed_sel)
#   - Writes:
#       (1) CSV with per-(p,blk) metrics: data/runs/compare_c3_psweep.csv
#       (2) One constellation .jld2 per p in: data/runs_comp3ways_constellations/
#           keys: s_jsdc_all, s_turbo_all, s_eq_all, meta(p=...)
#
# JSDC column behavior:
#   - By default, "JSDC" = JT (decode_jsdc_turbo_rounds) so you can replace JSDC with JT.
#   - If you want the old JSDC, run with: --use_plain_jsdc 1
#
# Run:
#   julia --project=. scripts/c3_psweep.jl
#   julia --project=. scripts/c3_psweep.jl --psweep "0.2,0.3,0.4,0.5" --corr 0.10 --nblk 225
#
# Notes (performance):
#   - Decodes each method once per block (no extra passes for plotting).
#   - Constellation points are capped per p (maxpts) to avoid RAM blowups.
#
# Fixes:
#   - Robustly extracts soft symbols for EQ even if it doesn't expose bit-LLRs,
#     by falling back to symbol-domain outputs (xhat/s_hat/y_eq/...).

using Random, Statistics, Printf
using SparseArrays
using DataFrames, CSV
using JLD2

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
ensure_linksim_loaded!()

include(joinpath(ROOT, "lib", "compare_3ways.jl"))

if !isdefined(Main, :TurboEQ)
    include(joinpath(ROOT, "lib", "TurboEQ.jl"))
end
using .TurboEQ

# JSDC-TURBO rounds
include(joinpath(ROOT, "lib", "jsdc_turbo_rounds.jl"))

# ----------------------------
# Metrics helpers
# ----------------------------
psr64_from_u256(u_hat::Vector{Int}, u_true::Vector{Int}) = sum([
    all(u_hat[64k+1:64k+64] .== u_true[64k+1:64k+64]) for k in 0:3
]) / 4.0
ber_u256(u_hat::Vector{Int}, u_true::Vector{Int}) = mean(u_hat .!= u_true)

function u256hat(x)
    if x isa NamedTuple
        return Vector{Int}(Int.(getproperty(x, :u256_hat)))
    elseif x isa AbstractDict
        if haskey(x, :u256_hat); return Vector{Int}(Int.(x[:u256_hat])) end
        if haskey(x, "u256_hat"); return Vector{Int}(Int.(x["u256_hat"])) end
        error("u256hat: cannot find u256_hat in Dict keys=$(collect(keys(x)))")
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

# ----------------------------
# CLI parsing
# ----------------------------
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

# ----------------------------
# Soft-symbol extraction (for constellation saving)
# ----------------------------
function _get_any(x, names::Vector{Symbol})
    if x isa NamedTuple
        for nm in names
            hasproperty(x, nm) && return getproperty(x, nm)
        end
    elseif x isa AbstractDict
        for nm in names
            haskey(x, nm) && return x[nm]           # Symbol keys
            ks = String(nm)
            haskey(x, ks) && return x[ks]           # String keys
        end
    end
    return nothing
end

function llr1024_to_softsym(llr::AbstractVector{<:Real}; clip::Float64=20.0)
    L = Float64.(llr)
    @inbounds for i in eachindex(L)
        if L[i] > clip
            L[i] = clip
        elseif L[i] < -clip
            L[i] = -clip
        end
    end
    m = tanh.(0.5 .* L)          # bit soft-magnetization
    mI = m[1:2:end]; mQ = m[2:2:end]
    return ComplexF64.(mI, mQ) ./ sqrt(2)
end

function m1024_to_softsym(m::AbstractVector{<:Real})
    mm = Float64.(m)
    mI = mm[1:2:end]; mQ = mm[2:2:end]
    return ComplexF64.(mI, mQ) ./ sqrt(2)
end

# Accept already-complex soft symbols (len≈512) and just return them
sym512_to_softsym(s::AbstractVector) = ComplexF64.(s)

function extract_softsym(out; prefer_m::Bool=false, llr_clip::Float64=20.0)
    # Prefer m_final if requested (JSDC/JT)
    if prefer_m
        m = _get_any(out, [:m_final, :m_outer, :m_post, :m_bits, :m])
        if m !== nothing
            return m1024_to_softsym(vec(m))
        end
    end

    # Try bit-LLR fields (Turbo/EQ sometimes)
    llr = _get_any(out, [
        :llr1024_post, :llr1024_ch, :llr1024,
        :L_post, :L_ch, :L_full, :L, :L_eq
    ])
    if llr !== nothing
        return llr1024_to_softsym(vec(llr); clip=llr_clip)
    end

    # Fallback: symbol-domain outputs (common for EQ pipelines)
    sym = _get_any(out, [
        :s_hat, :shat, :xhat_sym, :xhat, :x_hat,
        :y_eq, :yhat, :z_hat, :z,
        :x_mmse, :x_lmmse, :x_teq
    ])
    if sym !== nothing
        v = vec(sym)
        # If it’s complex, plot it
        if eltype(v) <: Complex || any(x -> x isa Complex, v)
            return sym512_to_softsym(v)
        end
    end

    return ComplexF64[]
end

function push_capped!(dst::Vector{ComplexF64}, src::Vector{ComplexF64}, maxpts::Int)
    isempty(src) && return
    length(dst) >= maxpts && return
    nroom = maxpts - length(dst)
    if length(src) <= nroom
        append!(dst, src)
    else
        step = max(1, floor(Int, length(src) / nroom))
        @inbounds for i in 1:step:length(src)
            push!(dst, src[i])
            length(dst) >= maxpts && break
        end
    end
end

ptag(p) = @sprintf("p%.3f", p)

# ----------------------------
# Main
# ----------------------------
function main()
    # Inputs / outputs
    dataset_ldpc = joinpath(DATA_DIR, "replayswap_qpsk_concat_256_512_1024_from_realdata_donorLS_h20_rho1e-2.jld2")
    dataset_rsc  = joinpath(DATA_DIR, "replayswap_qpsk_RSCconcat_256_512_1024_from_realdata_donorLS_h20_rho1e-2.jld2")

    outcsv   = joinpath(DATA_DIR, "runs", "compare_c3_psweep.csv")
    outdirC  = joinpath(DATA_DIR, "runs_comp3ways_constellations")

    # Selection
    corr_thr = 0.10
    use_nblk = 225
    seed_sel = 12648430

    # Sweep
    psweep = collect(0.0:0.1:0.5)

    # TurboEQ params
    niters  = 4
    damp    = 1.5
    M_eq    = 10
    σ2_init = 0.01

    # Plain JSDC baseline params (used only if --use_plain_jsdc 1)
    inner_pil_boost = 40.0
    alpha_out  = 1.5
    lampar_out = 1.0
    lampil_out = 0.5
    etaz_out   = 1e-2
    gamz_out   = 1.0e-3
    maxit_out  = 300

    # JT knobs (DEFAULT JSDC)
    use_plain_jsdc = false   # --use_plain_jsdc 1 to revert
    jt_rounds = 2
    jt_prior_w_in  = 0.6
    jt_prior_w_out = 1.0
    jt_prior_damp  = 0.3
    jt_prior_clip  = 8.0
    jt_outer_alpha = 1.5

    # (optional) JT hypers
    jt_lampar_in = 1.0
    jt_lampil_in = 3.0
    jt_etaz_in   = 1e-2
    jt_gamz_in   = 1e-3
    jt_maxit_in  = 300

    # Constellation saving knobs
    maxpts   = 150_000
    llr_clip = 20.0

    # Debug (prints EQ keys/fields once)
    debug_eq_keys = false

    # CLI (simple)
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a=="--dataset_ldpc"; i+=1; dataset_ldpc=ARGS[i]
        elseif a=="--dataset_rsc"; i+=1; dataset_rsc=ARGS[i]
        elseif a=="--outcsv"; i+=1; outcsv=ARGS[i]
        elseif a=="--outdirC"; i+=1; outdirC=ARGS[i]
        elseif a=="--corr"; i+=1; corr_thr=parse(Float64,ARGS[i])
        elseif a=="--nblk"; i+=1; use_nblk=parse(Int,ARGS[i])
        elseif a=="--seed_sel"; i+=1; seed_sel=parse(Int,ARGS[i])
        elseif a=="--psweep"; i+=1; psweep=parse_psweep(ARGS[i])
        elseif a=="--M_eq"; i+=1; M_eq=parse(Int,ARGS[i])
        elseif a=="--niters"; i+=1; niters=parse(Int,ARGS[i])
        elseif a=="--damp"; i+=1; damp=parse(Float64,ARGS[i])
        elseif a=="--sigma2"; i+=1; σ2_init=parse(Float64,ARGS[i])
        elseif a=="--maxpts"; i+=1; maxpts=parse(Int,ARGS[i])
        elseif a=="--llr_clip"; i+=1; llr_clip=parse(Float64,ARGS[i])
        elseif a=="--use_plain_jsdc"; i+=1; use_plain_jsdc = (parse(Int,ARGS[i]) != 0)
        elseif a=="--jt_rounds"; i+=1; jt_rounds=parse(Int,ARGS[i])
        elseif a=="--jt_prior_w_in"; i+=1; jt_prior_w_in=parse(Float64,ARGS[i])
        elseif a=="--jt_prior_w_out"; i+=1; jt_prior_w_out=parse(Float64,ARGS[i])
        elseif a=="--jt_prior_damp"; i+=1; jt_prior_damp=parse(Float64,ARGS[i])
        elseif a=="--jt_prior_clip"; i+=1; jt_prior_clip=parse(Float64,ARGS[i])
        elseif a=="--jt_outer_alpha"; i+=1; jt_outer_alpha=parse(Float64,ARGS[i])
        elseif a=="--debug_eq_keys"; i+=1; debug_eq_keys = (parse(Int,ARGS[i]) != 0)
        else
            error("Unknown arg: $a")
        end
        i += 1
    end

    mkpath(dirname(outcsv))
    mkpath(outdirC)

    # Load datasets once
    dl0 = load_dataset_any(dataset_ldpc)
    dr0 = load_dataset_any(dataset_rsc)

    nblk = size(dl0["y_qpsk_swapped"], 1)
    @assert size(dr0["y_qpsk_swapped"], 1) == nblk

    # Select blocks once (same for all p)
    eligible = findall((dl0["corr_donor"] .>= corr_thr) .& (dr0["corr_donor"] .>= corr_thr))
    rng = MersenneTwister(seed_sel)
    shuffle!(rng, eligible)
    blk_list = eligible[1:min(use_nblk, length(eligible))]
    isempty(blk_list) && error("No eligible blocks at corr_thr=$corr_thr")

    itlv_l = get_interleaver(dl0["meta_out"])
    itlv_r = get_interleaver(dr0["meta_out"])

    # Init codes once
    codeO, colsO, idrowsO, _ = initcode(k1, n1, npc_local)
    codeI, colsI, idrowsI, _ = initcode(k2, n2, npc_local)
    codeO.icols === nothing && (encode(codeO, zeros(Int, k1)); nothing)
    codeI.icols === nothing && (encode(codeI, zeros(Int, k2)); nothing)

    HO = get_H_sparse(codeO)
    HI = get_H_sparse(codeI)
    parityO = build_parity_indices(HO)
    parityI = build_parity_indices(HI)

    # Output table (keep column names stable)
    df = DataFrame(
        p=Float64[], blk=Int[], corr=Float64[],
        turbo_psr64=Float64[], turbo_ber=Float64[],
        eq_psr64=Float64[],    eq_ber=Float64[],
        jsdc_psr64=Float64[],  jsdc_ber=Float64[],
    )

    println("==============================================================")
    @printf("C3 PSWEEP | corr_thr=%.2f | using=%d blocks | JSDC=%s\n",
            corr_thr, length(blk_list), use_plain_jsdc ? "PLAIN" : "JT(default)")
    @printf("psweep = %s\n", string(psweep))
    println("==============================================================")

    printed_eq_keys = false

    for (pi, p) in enumerate(psweep)
        # per-p constellation buffers (capped)
        s_jsdc_all  = ComplexF64[]
        s_turbo_all = ComplexF64[]
        s_eq_all    = ComplexF64[]

        @printf("\n--- p=%.2f (%d/%d) ---\n", p, pi, length(psweep))

        for (ii, b) in enumerate(blk_list)
            # LDPC side
            yl = ComplexF64.(vec(dl0["y_qpsk_swapped"][b, :]))
            ul = Int.(vec(dl0["u256_mat"][b, :]))
            b512l  = Int.(vec(dl0["b512_mat"][b, :]))
            b512il = haskey(dl0, "b512_i_mat") ? Int.(vec(dl0["b512_i_mat"][b, :])) : b512l
            b1024l = Int.(vec(dl0["b1024_mat"][b, :]))
            hl = ComplexF64.(vec(dl0["h_blk_mat"][b, :]))

            # RSC side (TurboEQ)
            yr = ComplexF64.(vec(dr0["y_qpsk_swapped"][b, :]))
            ur = Int.(vec(dr0["u256_mat"][b, :]))
            b512r  = Int.(vec(dr0["b512_mat"][b, :]))
            b512ir = haskey(dr0, "b512_i_mat") ? Int.(vec(dr0["b512_i_mat"][b, :])) : b512r
            b1024r = Int.(vec(dr0["b1024_mat"][b, :]))
            hr = ComplexF64.(vec(dr0["h_blk_mat"][b, :]))

            # TurboEQ
            tb  = TurboEQ.decode_turboeq(yr, ur, b512r, b512ir, b1024r, hr, itlv_r;
                                         p=p, niters=niters, damp=damp, M_eq=M_eq, σ2_init=σ2_init)
            utb = u256hat(tb)
            tb_psr = psr64_from_u256(utb, ur)
            tb_ber = ber_u256(utb, ur)

            # EQ+SPA
            es = decode_eq_spa(yl, ul, b512l, b1024l, hl,
                               codeO, colsO, idrowsO, HO,
                               codeI, colsI, idrowsI, HI,
                               itlv_l; p=p, M_eq=M_eq)
            ues = u256hat(es)
            es_psr = psr64_from_u256(ues, ul)
            es_ber = ber_u256(ues, ul)

            if debug_eq_keys && !printed_eq_keys
                println("EQ typeof(es) = ", typeof(es))
                if es isa AbstractDict
                    println("EQ keys = ", collect(keys(es)))
                elseif es isa NamedTuple
                    println("EQ fields = ", propertynames(es))
                end
                printed_eq_keys = true
            end

            # "JSDC" (default = JT)
            js = nothing
            if use_plain_jsdc
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
            else
                js = decode_jsdc_turbo_rounds(
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
                    lampar_in=jt_lampar_in,
                    lampil_in=jt_lampil_in,
                    etaz_in=jt_etaz_in,
                    gamz_in=jt_gamz_in,
                    maxit_in=jt_maxit_in,
                    lampar_out=lampar_out,
                    lampil_out=lampil_out,
                    etaz_out=etaz_out,
                    gamz_out=gamz_out,
                    maxit_out=maxit_out
                )
            end

            ujs = u256hat(js)
            js_psr = psr64_from_u256(ujs, ul)
            js_ber = ber_u256(ujs, ul)

            # Constellation points (cheap + capped)
            push_capped!(s_jsdc_all,  extract_softsym(js; prefer_m=true,  llr_clip=llr_clip), maxpts)
            push_capped!(s_turbo_all, extract_softsym(tb; prefer_m=false, llr_clip=llr_clip), maxpts)
            push_capped!(s_eq_all,    extract_softsym(es; prefer_m=false, llr_clip=llr_clip), maxpts)

            # Row
            c = Float64(min(dl0["corr_donor"][b], dr0["corr_donor"][b]))
            push!(df, (p=p, blk=b, corr=c,
                       turbo_psr64=tb_psr, turbo_ber=tb_ber,
                       eq_psr64=es_psr, eq_ber=es_ber,
                       jsdc_psr64=js_psr, jsdc_ber=js_ber))

            if ii == 1 || ii % 25 == 0 || ii == length(blk_list)
                @printf("  blk %d/%d | PSR64: T=%.3f EQ=%.3f J=%.3f\n",
                        ii, length(blk_list), tb_psr, es_psr, js_psr)
            end
        end

        # Save per-p constellation file (exact keys your grid plotter expects)
        meta = (p=p,
                corr_thr=corr_thr,
                seed_sel=seed_sel,
                nblk=length(blk_list),
                maxpts=maxpts,
                M_eq=M_eq,
                note="keys: s_jsdc_all/s_turbo_all/s_eq_all")

        outf = joinpath(outdirC, "c3const__$(ptag(p))__corr$(round(corr_thr,digits=2))__n$(length(blk_list)).jld2")
        @save outf s_jsdc_all s_turbo_all s_eq_all meta
        println("Saved constell → $outf")

        # Per-p summary (mean over blocks)
        sub = df[df.p .== p, :]
        @printf("  mean PSR64 @ p=%.2f: Turbo=%.3f EQ=%.3f JSDC=%.3f\n",
                p, mean(sub.turbo_psr64), mean(sub.eq_psr64), mean(sub.jsdc_psr64))
    end

    # Save CSV once
    CSV.write(outcsv, df)
    println("\nSaved CSV → $outcsv")
    println("Done.")
end

main()
