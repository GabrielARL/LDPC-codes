#!/usr/bin/env julia
# scripts/make_aug_from_realdata.jl
#
# Build augmented datasets (LDPC + RSC) from REALDATA replay-swap base datasets.
#
# Augmentation model (per selected base block b0):
#   y_aug = yhat(base) + alpha * e_tilde(donor) + sigw * w
# where:
#   yhat(base) = conv_prefix(h_base, s_true(base))
#   e_tilde(donor) is residual bank from SAME dataset (LDPC residuals for LDPC, RSC for RSC)
#   w ~ CN(0,1) iid (complex AWGN), so sigw is the complex std dev
#
# Output:
#   data/auggrid/LDPC__AUG__...__aX__sigwY.jld2
#   data/auggrid/RSC__AUG__...__aX__sigwY.jld2
#
# Example:
#   julia --project=. scripts/make_aug_from_realdata.jl \
#     --alphas "0.5,1.0,1.5" --sigw "0.0,0.02" \
#     --K 5 --mode shuffle --corr 0.80 --nblk 6 \
#     --seed_sel 12648430 --seed_aug 123 \
#     --outdir data/auggrid
#
# Notes:
# - Defaults assume base datasets live in DFEC/data/ (or DFEC/data/raw/; see below).
# - Requires dataset keys: y_qpsk_swapped, b1024_mat, h_blk_mat, corr_donor
#   and typically u256_mat, b512_mat, b1024_mat, h_blk_mat, meta_out

using Random, Printf, Statistics, LinearAlgebra
using JLD2

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
include(joinpath(ROOT, "lib", "ModemQPSK.jl"))
using .ModemQPSK: qpsk_from_bits

# ---------------------------
# Helpers
# ---------------------------
@inline _get_blockvec(A, b::Int) = vec(A[b, :])

function load_dataset_any(path::String)
    isfile(path) || error("Dataset file not found: $path")
    d = JLD2.load(path)
    if haskey(d, "data") && (d["data"] isa AbstractDict)
        return Dict{String,Any}(d["data"])
    else
        return Dict{String,Any}(d)
    end
end

function needkey(d::AbstractDict, k::String, path::String)
    haskey(d, k) || error("Dataset missing key \"$k\" in: $path\nKeys present: $(collect(keys(d)))")
    return d[k]
end

function parse_list_float(s::String)
    ss = replace(strip(s), " " => "")
    isempty(ss) && return Float64[]
    return Float64.(parse.(Float64, split(ss, ",")))
end
parse_int(s::String)   = parse(Int, strip(s))
parse_float(s::String) = parse(Float64, strip(s))

# Prefix convolution: y[t] = sum_{ℓ<=t} h[ℓ] * x[t-ℓ+1]
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

# Compute residual bank e[b,:] = y - conv_prefix(h, s_true)
function compute_residuals(d::Dict{String,Any}; Lh_max::Int=20)
    ymat   = d["y_qpsk_swapped"]
    b1024m = d["b1024_mat"]
    hmat   = d["h_blk_mat"]

    nblk = size(ymat, 1)
    T    = size(ymat, 2)
    e_mat = Matrix{ComplexF64}(undef, nblk, T)

    @info "Computing residual bank e = y - yhat" nblk T Lh_max

    @inbounds for b in 1:nblk
        y     = ComplexF64.(_get_blockvec(ymat, b))
        b1024 = Int.(_get_blockvec(b1024m, b))
        s_true = ComplexF64.(qpsk_from_bits(b1024))   # 512 symbols for 1024 bits
        hfull  = ComplexF64.(_get_blockvec(hmat, b))

        Lh = min(Lh_max, length(hfull), T)
        h = hfull[1:Lh]
        yhat = conv_prefix(h, s_true, T)
        e_mat[b, :] .= y .- yhat
    end

    return e_mat
end

# Build augmented dict for ONLY selected blocks; output has nsel*K blocks
function make_augmented_subset(d0::Dict{String,Any}, blk_list::Vector{Int};
                              K::Int, alpha::Float64, sigw::Float64, mode::String,
                              seed_aug::Int, Lh_max::Int=20)

    for k in ("y_qpsk_swapped", "b1024_mat", "h_blk_mat")
        haskey(d0, k) || error("Missing key: $k")
    end

    ymat   = d0["y_qpsk_swapped"]
    b1024m = d0["b1024_mat"]
    hmat   = d0["h_blk_mat"]

    T    = size(ymat, 2)
    nsel = length(blk_list)
    rng  = MersenneTwister(seed_aug)

    e_mat = compute_residuals(d0; Lh_max=Lh_max)
    nblk_all = size(e_mat, 1)

    y_aug = Matrix{ComplexF64}(undef, nsel*K, T)

    donor_idx = Vector{Int}(undef, nsel*K)
    base_blk  = Vector{Int}(undef, nsel*K)
    aug_rep   = Vector{Int}(undef, nsel*K)

    @info "Augmenting selected blocks" nsel K alpha sigw mode seed_aug

    # CN(0,1): (randn + i*randn)/sqrt(2)
    @inline function cn1!(buf::Vector{ComplexF64})
        @inbounds for i in eachindex(buf)
            buf[i] = (randn(rng) + 1im*randn(rng)) / sqrt(2)
        end
        return buf
    end
    wbuf = zeros(ComplexF64, T)

    @inbounds for kk in 1:K
        for (ii, b0) in enumerate(blk_list)
            b1024 = Int.(_get_blockvec(b1024m, b0))
            s_true = ComplexF64.(qpsk_from_bits(b1024))
            hfull  = ComplexF64.(_get_blockvec(hmat, b0))

            Lh = min(Lh_max, length(hfull), T)
            h = hfull[1:Lh]
            yhat = conv_prefix(h, s_true, T)

            idx = if mode == "same"
                b0
            elseif mode == "shuffle"
                rand(rng, 1:nblk_all)
            elseif mode == "cyclic"
                ((b0 + (kk-1) - 1) % nblk_all) + 1
            else
                error("Unknown --mode=$mode (use same|shuffle|cyclic)")
            end

            row = (kk-1)*nsel + ii

            # yhat + alpha*e_tilde
            y_aug[row, :] .= yhat .+ alpha .* view(e_mat, idx, :)

            # + sigw * w
            if sigw > 0
                cn1!(wbuf)
                y_aug[row, :] .+= sigw .* wbuf
            end

            donor_idx[row] = idx
            base_blk[row]  = b0
            aug_rep[row]   = kk
        end
    end

    # replicate per-block matrices
    function rep_rows(A)
        B = similar(A, nsel*K, size(A,2))
        @inbounds for kk in 1:K
            for (ii, b0) in enumerate(blk_list)
                row = (kk-1)*nsel + ii
                B[row, :] .= A[b0, :]
            end
        end
        return B
    end

    # replicate per-block vectors
    function rep_vec(v::AbstractVector)
        B = Vector{eltype(v)}(undef, nsel*K)
        @inbounds for kk in 1:K
            for (ii, b0) in enumerate(blk_list)
                row = (kk-1)*nsel + ii
                B[row] = v[b0]
            end
        end
        return B
    end

    out = Dict{String,Any}()
    out["y_qpsk_swapped"] = y_aug

    for key in ("u256_mat", "b512_mat", "b512_i_mat", "b1024_mat", "h_blk_mat")
        haskey(d0, key) && (out[key] = rep_rows(d0[key]))
    end

    for key in ("corr_donor", "corr_post", "sigma2_hat")
        if haskey(d0, key) && (d0[key] isa AbstractVector)
            out[key] = rep_vec(d0[key])
        end
    end

    out["aug_base_blk"]  = base_blk
    out["aug_rep"]       = aug_rep
    out["aug_donor_idx"] = donor_idx

    # copy other fields as-is
    for (k,v) in d0
        haskey(out, k) || (out[k] = v)
    end

    # attach augmentation info into meta_out if present
    if haskey(out, "meta_out") && (out["meta_out"] isa NamedTuple)
        meta0 = out["meta_out"]
        out["meta_out"] = (;
            meta0...,
            augmentation = (;
                note="yhat(base) + alpha*residual(donor) + sigw*CN(0,1)",
                K=K, alpha=alpha, sigw=sigw, mode=mode, seed_aug=seed_aug, Lh_max=Lh_max,
                nsel=length(blk_list)
            )
        )
    end

    return out
end

# ---------------------------
# Main / CLI
# ---------------------------
function cli_main()
    # Defaults: try DFEC/data first; if not there, you can point to data/raw or pass full paths.
    dataset_ldpc = joinpath(DATA_DIR, "replayswap_qpsk_concat_256_512_1024_from_realdata_donorLS_h20_rho1e-2.jld2")
    dataset_rsc  = joinpath(DATA_DIR, "replayswap_qpsk_RSCconcat_256_512_1024_from_realdata_donorLS_h20_rho1e-2.jld2")

    # If you store them in data/raw/, uncomment these two:
    # dataset_ldpc = joinpath(DATA_DIR, "raw", basename(dataset_ldpc))
    # dataset_rsc  = joinpath(DATA_DIR, "raw", basename(dataset_rsc))

    outdir = joinpath(DATA_DIR, "auggrid")

    alphas = [1.0]
    sigws  = [0.0]
    corr_thr = 0.80
    use_nblk = 6         # -1 = all eligible
    K = 5
    mode = "shuffle"     # same|shuffle|cyclic
    seed_sel = 12648430
    seed_aug = 123
    Lh_max = 20

    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--alphas"; i+=1; alphas = parse_list_float(ARGS[i])
        elseif a == "--sigw" || a == "--sigws"; i+=1; sigws = parse_list_float(ARGS[i])
        elseif a == "--corr"; i+=1; corr_thr = parse_float(ARGS[i])
        elseif a == "--nblk"; i+=1; use_nblk = parse_int(ARGS[i])
        elseif a == "--K"; i+=1; K = parse_int(ARGS[i])
        elseif a == "--mode"; i+=1; mode = ARGS[i]
        elseif a == "--seed_sel"; i+=1; seed_sel = parse_int(ARGS[i])
        elseif a == "--seed_aug"; i+=1; seed_aug = parse_int(ARGS[i])
        elseif a == "--Lhmax"; i+=1; Lh_max = parse_int(ARGS[i])
        elseif a == "--dataset_ldpc"; i+=1; dataset_ldpc = ARGS[i]
        elseif a == "--dataset_rsc";  i+=1; dataset_rsc  = ARGS[i]
        elseif a == "--outdir"; i+=1; outdir = ARGS[i]
        elseif a == "--help" || a == "-h"
            println("""
Usage:
  julia --project=. scripts/make_aug_from_realdata.jl [args]

Grid:
  --alphas "0.5,1.0,1.5"
  --sigw   "0.0,0.02,0.05"      (complex AWGN std dev)

Selection:
  --corr 0.80
  --nblk 6                     (-1 = all eligible)
  --seed_sel 12648430

Augmentation:
  --K 5
  --mode same|shuffle|cyclic
  --seed_aug 123
  --Lhmax 20

Paths:
  --dataset_ldpc <path>
  --dataset_rsc  <path>
  --outdir <dir>               (default: data/auggrid)
""")
            return
        else
            error("Unknown arg: $a")
        end
        i += 1
    end

    isfile(dataset_ldpc) || error("LDPC dataset not found: $dataset_ldpc")
    isfile(dataset_rsc)  || error("RSC dataset not found:  $dataset_rsc")
    mkpath(outdir)

    dl0 = load_dataset_any(dataset_ldpc)
    dr0 = load_dataset_any(dataset_rsc)

    needkey(dl0, "y_qpsk_swapped", dataset_ldpc)
    needkey(dl0, "corr_donor", dataset_ldpc)
    needkey(dr0, "y_qpsk_swapped", dataset_rsc)
    needkey(dr0, "corr_donor", dataset_rsc)

    nblk = size(dl0["y_qpsk_swapped"], 1)
    @assert size(dr0["y_qpsk_swapped"], 1) == nblk "Block count mismatch between LDPC and RSC datasets"

    eligible = findall((dl0["corr_donor"] .>= corr_thr) .& (dr0["corr_donor"] .>= corr_thr))
    rng = MersenneTwister(seed_sel)
    shuffle!(rng, eligible)
    blk_list = (use_nblk < 0) ? eligible : eligible[1:min(use_nblk, length(eligible))]
    isempty(blk_list) && error("No eligible blocks at corr_thr=$corr_thr")

    nsel = length(blk_list)
    @info "Eligible base blocks" total=length(eligible) n_used=nsel corr_thr=corr_thr
    @printf("\nBase blocks selected: %d | K=%d => augmented rows=%d\n", nsel, K, nsel*K)
    @printf("alphas=%s\n", string(alphas))
    @printf("sigw  =%s\n\n", string(sigws))

    for a in alphas, sw in sigws
        tag = @sprintf("__K%d__mode%s__corr%.2f__nsel%d__seedSel%d__seedAug%d__a%.3f__sigw%.5f__Lh%d",
                       K, mode, corr_thr, nsel, seed_sel, seed_aug, a, sw, Lh_max)

        out_ldpc = joinpath(outdir, "LDPC__AUG" * tag * ".jld2")
        out_rsc  = joinpath(outdir, "RSC__AUG"  * tag * ".jld2")

        println("Building: alpha=$(a) sigw=$(sw)")
        dl = make_augmented_subset(copy(dl0), blk_list; K=K, alpha=a, sigw=sw, mode=mode, seed_aug=seed_aug, Lh_max=Lh_max)
        dr = make_augmented_subset(copy(dr0), blk_list; K=K, alpha=a, sigw=sw, mode=mode, seed_aug=seed_aug, Lh_max=Lh_max)

        meta = (
            base_blk_list = Vector{Int}(blk_list),
            corr_thr = corr_thr,
            K = K, alpha = a, sigw = sw, mode = mode,
            seed_sel = seed_sel, seed_aug = seed_aug,
            dataset_ldpc = dataset_ldpc,
            dataset_rsc  = dataset_rsc,
            Lh_max = Lh_max
        )

        @save out_ldpc dl meta
        @save out_rsc  dr meta

        println("  saved → $out_ldpc")
        println("  saved → $out_rsc")
    end

    println("\nDone.")
end

cli_main()
