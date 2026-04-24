#!/usr/bin/env julia
# scripts/debug_eq_constellation.jl
#
# Debug helper: verify why EQ constellations are empty.
# It runs ONE block decode and prints:
#   - type + keys/fields returned by decode_eq_spa
#   - which key (if any) we can use to extract constellation points
#   - how many points we extract and their basic stats
#
# Usage:
#   julia --project=. scripts/debug_eq_constellation.jl
#   julia --project=. scripts/debug_eq_constellation.jl --p 0.30 --corr 0.10 --nblk 225 --blk 1
#   julia --project=. scripts/debug_eq_constellation.jl --blk 50 --p 0.20
#
using Random, Statistics, Printf
using SparseArrays

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
ensure_linksim_loaded!()
include(joinpath(ROOT, "lib", "compare_3ways.jl"))

# ----------------------------
# small helpers
# ----------------------------
function build_parity_indices(H::SparseMatrixCSC{Bool, Int})
    m, _n = size(H)
    pi = [Int[] for _ in 1:m]
    I, J, _ = findnz(H)
    @inbounds for (i, j) in zip(I, J)
        push!(pi[i], j)
    end
    return pi
end

function _get_any(x, names::Vector{Symbol})
    if x isa NamedTuple
        for nm in names
            hasproperty(x, nm) && return getproperty(x, nm)
        end
    elseif x isa AbstractDict
        for nm in names
            haskey(x, nm) && return x[nm]
            ks = String(nm)
            haskey(x, ks) && return x[ks]
        end
    end
    return nothing
end

# Convert whatever "symbol-like" thing to ComplexF64[.] for plotting
function sym_any_to_softsym(sym)
    sym === nothing && return ComplexF64[]

    # Vector
    if sym isa AbstractVector
        v = vec(sym)
        # already complex?
        if eltype(v) <: Complex || any(x -> x isa Complex, v)
            return ComplexF64.(v)
        end
        # interleaved real I/Q
        if length(v) % 2 == 0
            I = Float64.(v[1:2:end])
            Q = Float64.(v[2:2:end])
            return ComplexF64.(I, Q) ./ sqrt(2)
        end
        return ComplexF64[]
    end

    # Matrix 512×2 or 2×512
    if sym isa AbstractMatrix
        A = Float64.(sym)
        if size(A,2) == 2
            return ComplexF64.(A[:,1], A[:,2]) ./ sqrt(2)
        elseif size(A,1) == 2
            return ComplexF64.(A[1,:], A[2,:]) ./ sqrt(2)
        end
        return ComplexF64[]
    end

    return ComplexF64[]
end

function summarize_points(name::String, s::Vector{ComplexF64}; maxshow::Int=3)
    println("[$name] npts = ", length(s))
    isempty(s) && return
    r = real.(s); q = imag.(s)
    @printf("  I: mean=%.4f std=%.4f min=%.4f max=%.4f\n", mean(r), std(r), minimum(r), maximum(r))
    @printf("  Q: mean=%.4f std=%.4f min=%.4f max=%.4f\n", mean(q), std(q), minimum(q), maximum(q))
    println("  head = ", s[1:min(end, maxshow)])
end

# ----------------------------
# main
# ----------------------------
function main()
    dataset_ldpc = joinpath(DATA_DIR, "replayswap_qpsk_concat_256_512_1024_from_realdata_donorLS_h20_rho1e-2.jld2")
    corr_thr = 0.10
    use_nblk = 225
    seed_sel = 12648430
    p = 0.30
    blk_idx = 1          # index within the selected block list, not absolute block id
    M_eq = 6

    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a=="--dataset_ldpc"; i+=1; dataset_ldpc=ARGS[i]
        elseif a=="--corr"; i+=1; corr_thr=parse(Float64,ARGS[i])
        elseif a=="--nblk"; i+=1; use_nblk=parse(Int,ARGS[i])
        elseif a=="--seed_sel"; i+=1; seed_sel=parse(Int,ARGS[i])
        elseif a=="--p"; i+=1; p=parse(Float64,ARGS[i])
        elseif a=="--blk"; i+=1; blk_idx=parse(Int,ARGS[i])
        elseif a=="--M_eq"; i+=1; M_eq=parse(Int,ARGS[i])
        else
            error("Unknown arg: $a")
        end
        i += 1
    end

    dl0 = load_dataset_any(dataset_ldpc)
    nblk_total = size(dl0["y_qpsk_swapped"], 1)

    eligible = findall(dl0["corr_donor"] .>= corr_thr)
    rng = MersenneTwister(seed_sel)
    shuffle!(rng, eligible)
    blk_list = eligible[1:min(use_nblk, length(eligible))]
    isempty(blk_list) && error("No eligible blocks at corr_thr=$corr_thr")

    blk_idx = clamp(blk_idx, 1, length(blk_list))
    b = blk_list[blk_idx]
    @printf("Selected block: blk_idx=%d -> b=%d (of total %d)\n", blk_idx, b, nblk_total)

    itlv_l = get_interleaver(dl0["meta_out"])

    # init codes once
    codeO, colsO, idrowsO, _ = initcode(k1, n1, npc_local)
    codeI, colsI, idrowsI, _ = initcode(k2, n2, npc_local)
    codeO.icols === nothing && (encode(codeO, zeros(Int, k1)); nothing)
    codeI.icols === nothing && (encode(codeI, zeros(Int, k2)); nothing)

    HO = get_H_sparse(codeO)
    HI = get_H_sparse(codeI)
    parityO = build_parity_indices(HO)
    parityI = build_parity_indices(HI)

    # pull block arrays (LDPC side)
    yl = ComplexF64.(vec(dl0["y_qpsk_swapped"][b, :]))
    ul = Int.(vec(dl0["u256_mat"][b, :]))
    b512l  = Int.(vec(dl0["b512_mat"][b, :]))
    b1024l = Int.(vec(dl0["b1024_mat"][b, :]))
    hl = ComplexF64.(vec(dl0["h_blk_mat"][b, :]))

    println("Running decode_eq_spa ...")
    es = decode_eq_spa(yl, ul, b512l, b1024l, hl,
                       codeO, colsO, idrowsO, HO,
                       codeI, colsI, idrowsI, HI,
                       itlv_l; p=p, M_eq=M_eq)

    println("\n==== EQ output inspection ====")
    println("typeof(es) = ", typeof(es))
    if es isa AbstractDict
        println("Dict keys = ", collect(keys(es)))
    elseif es isa NamedTuple
        println("NamedTuple fields = ", propertynames(es))
    end

    # Try to find LLR-ish keys
    llr = _get_any(es, [:L_full, :L_ch, :L_post, :L, :L_eq, :llr1024, :llr1024_ch, :llr1024_post])
    println("\nFound llr key? ", llr === nothing ? "NO" : "YES")
    if llr !== nothing
        v = vec(llr)
        println("  llr length = ", length(v), " eltype=", eltype(v))
        println("  llr head = ", v[1:min(end,6)])
    end

    # Try to find symbol-ish keys (these are the usual culprits)
    sym = _get_any(es, [:xhat_sym, :s_hat, :xhat, :x_hat, :y_eq, :z_hat, :z, :x_mmse, :x_lmmse, :x_teq])
    println("\nFound symbol key? ", sym === nothing ? "NO" : "YES")
    if sym !== nothing
        println("  sym typeof = ", typeof(sym), " size/len = ",
                sym isa AbstractMatrix ? string(size(sym)) : string(length(vec(sym))))
        if sym isa AbstractVector
            println("  sym head = ", vec(sym)[1:min(end,6)])
        elseif sym isa AbstractMatrix
            println("  sym head row1 = ", sym[1, 1:min(end,6)])
        end
    end

    # Convert to points and summarize
    s = sym_any_to_softsym(sym)
    println("\n==== Extracted EQ constellation points ====")
    summarize_points("EQ", s)

    # Also: if we found LLRs, show how many points we'd get from LLR->tanh
    if llr !== nothing
        L = Float64.(vec(llr))
        m = tanh.(0.5 .* L)
        s_llr = ComplexF64.(m[1:2:end], m[2:2:end]) ./ sqrt(2)
        summarize_points("EQ_from_LLR", s_llr)
    end

    println("\nIf EQ points are still empty, copy/paste the printed keys/fields here.")
end

main()
