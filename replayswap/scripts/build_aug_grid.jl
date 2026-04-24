#!/usr/bin/env julia
# scripts/build_aug_grid.jl
#
# Build augmented datasets (LDPC + RSC) for multiple alpha and sigw values.

using Random, Printf, Statistics, LinearAlgebra
using JLD2

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
include(joinpath(ROOT, "lib", "ModemQPSK.jl"))
using .ModemQPSK: qpsk_from_bits

@inline _get_blockvec(A, b::Int) = vec(A[b, :])

function needkey(d::AbstractDict, k::String, path::String)
    haskey(d, k) || error("Dataset missing key \"$k\" in: $path\nKeys present: $(collect(keys(d)))")
    return d[k]
end

function load_dataset_any(path::String)
    isfile(path) || error("Dataset file not found: $path")
    d = JLD2.load(path)
    if haskey(d, "data") && (d["data"] isa AbstractDict)
        return Dict{String,Any}(d["data"])
    else
        return Dict{String,Any}(d)
    end
end

function parse_list_float(s::String)
    ss = strip(s)
    isempty(ss) && error("Empty list string")
    return Float64.(parse.(Float64, split(replace(ss, " "=>""), ",")))
end
parse_int(s::String)   = parse(Int, strip(s))
parse_float(s::String) = parse(Float64, strip(s))

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

function compute_residuals(d; Lh_max::Int=20)
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
        s_true = ComplexF64.(qpsk_from_bits(b1024))
        hfull  = ComplexF64.(_get_blockvec(hmat, b))

        Lh = min(Lh_max, length(hfull), T)
        h = hfull[1:Lh]
        yhat = conv_prefix(h, s_true, T)
        e_mat[b, :] .= y .- yhat
    end
    return e_mat
end

function make_augmented_subset(d::Dict{String,Any}, blk_list::Vector{Int};
                              K::Int, alpha::Float64, sigw::Float64, mode::String,
                              seed_aug::Int, Lh_max::Int=20)

    for k in ("y_qpsk_swapped", "b1024_mat", "h_blk_mat")
        haskey(d, k) || error("Missing key: $k")
    end

    ymat   = d["y_qpsk_swapped"]
    b1024m = d["b1024_mat"]
    hmat   = d["h_blk_mat"]

    T    = size(ymat, 2)
    nsel = length(blk_list)
    rng  = MersenneTwister(seed_aug)

    e_mat = compute_residuals(d; Lh_max=Lh_max)
    nblk_all = size(e_mat, 1)

    y_aug = Matrix{ComplexF64}(undef, nsel*K, T)
    donor_idx = Vector{Int}(undef, nsel*K)
    base_blk  = Vector{Int}(undef, nsel*K)
    aug_rep   = Vector{Int}(undef, nsel*K)

    @info "Augmenting selected blocks" nsel K alpha sigw mode seed_aug

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
            y_aug[row, :] .= yhat .+ alpha .* view(e_mat, idx, :)

            if sigw > 0
                cn1!(wbuf)
                y_aug[row, :] .+= sigw .* wbuf
            end

            donor_idx[row] = idx
            base_blk[row]  = b0
            aug_rep[row]   = kk
        end
    end

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
        haskey(d, key) && (out[key] = rep_rows(d[key]))
    end

    for key in ("corr_donor", "corr_post", "sigma2_hat")
        if haskey(d, key) && (d[key] isa AbstractVector)
            out[key] = rep_vec(d[key])
        end
    end

    out["aug_base_blk"]  = base_blk
    out["aug_rep"]       = aug_rep
    out["aug_donor_idx"] = donor_idx

    for (k,v) in d
        haskey(out, k) || (out[k] = v)
    end

    if haskey(out, "meta_out") && (out["meta_out"] isa NamedTuple)
        meta0 = out["meta_out"]
        out["meta_out"] = (;
            meta0...,
            augmentation = (;
                note="yhat(base) + alpha*residual(donor) + sigw*CN(0,1)",
                K=K, alpha=alpha, sigw=sigw, mode=mode, seed_aug=seed_aug, Lh_max=Lh_max,
                nsel=nsel
            )
        )
    end

    return out
end

function main()
    # defaults (edit as needed)
    dataset_rsc  = joinpath(DATA_DIR, "replayswap_qpsk_RSCconcat_256_512_1024_from_realdata_donorLS_h20_rho1e-2.jld2")
    dataset_ldpc = joinpath(DATA_DIR, "replayswap_qpsk_concat_256_512_1024_from_realdata_donorLS_h20_rho1e-2.jld2")
    outdir = joinpath(DATA_DIR, "auggrid")

    alphas = [1.0]
    sigws  = [0.0]
    corr_thr = 0.80
    use_nblk = 6
    K = 5
    mode = "shuffle"
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
            println("Run: julia --project=. scripts/build_aug_grid.jl --alphas \"0.5,1.0\" --sigw \"0.0,0.02\" --corr 0.80 --nblk 6 --K 5 --mode shuffle --outdir data/auggrid")
            return
        else
            error("Unknown arg: $a")
        end
        i += 1
    end

    mkpath(outdir)

    dl0 = load_dataset_any(dataset_ldpc)
    dr0 = load_dataset_any(dataset_rsc)

    needkey(dl0, "y_qpsk_swapped", dataset_ldpc)
    needkey(dl0, "corr_donor", dataset_ldpc)
    needkey(dr0, "y_qpsk_swapped", dataset_rsc)
    needkey(dr0, "corr_donor", dataset_rsc)

    nblk = size(dl0["y_qpsk_swapped"], 1)
    @assert size(dr0["y_qpsk_swapped"], 1) == nblk

    eligible = findall((dl0["corr_donor"] .>= corr_thr) .& (dr0["corr_donor"] .>= corr_thr))
    rng = MersenneTwister(seed_sel)
    shuffle!(rng, eligible)
    blk_list = (use_nblk < 0) ? eligible : eligible[1:min(use_nblk, length(eligible))]
    isempty(blk_list) && error("No eligible blocks at corr_thr=$corr_thr")

    nsel = length(blk_list)
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

main()

