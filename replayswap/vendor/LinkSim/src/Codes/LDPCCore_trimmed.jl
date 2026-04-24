# src/Codes/LDPCCore_trimmed.jl
#
# TRIMMED LDPC core for DFEC / lean LinkSim:
#   - No GaloisFields, StatsBase, SignalAnalysis
#   - Keeps: Code, ldpc_paths, readgenerator, readsparse,
#            FEC_create_code, initcode, encode,
#            get_H_sparse, sum_product_decode, is_valid_codeword,
#            (optional) modulate/demodulate
#
# Assumes LDPC files exist at:
#   codes/ldpc/<k>-<n>-<npc>.{H,gen,pchk}
# or ENV["LDPC_PATH"] points to that folder.

using SparseArrays
using Random
using LinearAlgebra
using DelimitedFiles

# =========================
# Types & Pretty Print
# =========================

mutable struct Code
    k::Int
    n::Int
    npc::Int
    icols::Union{Nothing, Vector{Int}}     # permutation for systematic ordering
    gen::Union{Nothing, BitMatrix}         # dense generator chunks (from .gen)
    H::Matrix{Bool}                        # dense Bool for convenience; sparse view cached
end

Base.show(io::IO, ldpc::Code) = print(io, "LDPC($(ldpc.k)/$(ldpc.n))")

# =========================
# Optional modulation helpers (not used by SPA/JSDC; safe to keep)
# =========================
modulate(x; θ=0.0)   = x == 1 ? cis(θ) : -cis(θ)
demodulate(x; θ=0.0) = x == 1 ? cis(θ) : -cis(θ)

# =========================
# H sparse cache
# =========================
const H_sparse_cache = IdDict{Code, SparseMatrixCSC{Bool, Int}}()

get_H_sparse(code::Code) = get!(H_sparse_cache, code) do
    sparse(code.H)
end

# =========================
# LDPC file paths
# =========================
function ldpc_paths(k::Int, n::Int, npc::Int=4)
    basecands = String[]
    haskey(ENV, "LDPC_PATH") && push!(basecands, ENV["LDPC_PATH"])
    push!(basecands, joinpath(pwd(), "codes", "ldpc"))
    push!(basecands, normpath(joinpath(@__DIR__, "..", "..", "codes", "ldpc")))

    stem = "$(k)-$(n)-$(npc)"
    for base in basecands
        pH   = joinpath(base, stem * ".H")
        pgen = joinpath(base, stem * ".gen")
        ppch = joinpath(base, stem * ".pchk")
        if isfile(pH) || isfile(pgen) || isfile(ppch)
            return (isfile(pH)   ? pH   : nothing,
                    isfile(pgen) ? pgen : nothing,
                    isfile(ppch) ? ppch : nothing)
        end
    end
    error("LDPC files not found for $(stem). Checked: " * join(basecands, ", "))
end

# Optional: load a dense H from a whitespace matrix file (not your .H adjacency format)
function load_H_bool(path_H::AbstractString)
    A = readdlm(path_H)
    m, n = size(A)
    rows = Int[]; cols = Int[]
    @inbounds for i in 1:m, j in 1:n
        if A[i,j] != 0
            push!(rows, i); push!(cols, j)
        end
    end
    return sparse(rows, cols, trues(length(rows)), m, n)
end

# =========================
# File readers (.gen / .H adjacency list)
# =========================

# Your .gen reader (dense generator, with column permutation)
function readgenerator(filename::AbstractString)
    open(filename) do io
        read(io, UInt32) == 0x00004780 || error("Bad generator")
        read(io, UInt8)  == 0x64       || error("Bad generator: must be dense")

        p = Int(read(io, UInt32))
        n = Int(read(io, UInt32))

        # stored as 0-based; invperm to get "position -> original index"
        icols = [Int(read(io, UInt32)) + 1 for _ in 1:n] |> invperm

        Int(read(io, UInt32)) == p     || error("Bad row size")
        Int(read(io, UInt32)) == n - p || error("Bad column size")

        G = mapreduce(hcat, 1:(n - p)) do _
            v = [read(io, UInt32) for _ in 1:ceil(Int, p / 32)]
            isodd(length(v)) && push!(v, 0)
            b = BitArray(undef, p)
            b.chunks .= reinterpret(UInt64, v)
            b
        end
        return icols, BitMatrix(G)
    end
end

# Your .H reader: adjacency list style:
# line: "row: col1 col2 col3 ..."
function readsparse(filename::AbstractString)
    ii, jj = Int[], Int[]
    for s in readlines(filename)
        m = match(r"^ *(\d+):(.+)$", s)
        if m !== nothing
            i  = parse(Int, m[1])
            js = parse.(Int, split(m[2]))
            append!(ii, repeat([i + 1], length(js)))
            append!(jj, js .+ 1)
        end
    end
    return sparse(ii, jj, true)
end

# =========================
# Code construction + encoding
# =========================

function FEC_create_code(k::Int, n::Int, npc::Int)
    pH, _, _ = ldpc_paths(k, n, npc)
    pH === nothing && error("Missing .H file for LDPC($(k)/$(n)) npc=$(npc)")
    Hs = readsparse(pH)
    return Code(k, n, npc, nothing, nothing, collect(Hs))
end

# Encode:
# - load icols + generator if needed
# - compute parity bits via XOR over positions where gen row has 1 and bits is 1
# - pack [parity; bits] then permute by icols
function encode(ldpc::Code, bits::AbstractVector{<:Integer})
    length(bits) == ldpc.k || throw(ArgumentError("Wrong bit length: got $(length(bits)) expected $(ldpc.k)"))

    if ldpc.icols === nothing || ldpc.gen === nothing
        _, pgen, _ = ldpc_paths(ldpc.k, ldpc.n, ldpc.npc)
        pgen === nothing && error("Missing .gen file for LDPC($(ldpc.k)/$(ldpc.n)) npc=$(ldpc.npc)")
        ldpc.icols, ldpc.gen = readgenerator(pgen)
    end

    G = ldpc.gen
    p = size(G, 1)
    parity = BitVector(undef, p)

    @inbounds for r in 1:p
        acc = 0
        for j in 1:ldpc.k
            # generator row is BitMatrix => G[r,j] is Bool
            if G[r, j] && (bits[j] & 0x1) == 1
                acc ⊻= 1
            end
        end
        parity[r] = (acc == 1)
    end

    cw = vcat(parity, BitVector(bits .!= 0))
    return cw[ldpc.icols]
end

# =========================
# Neighbor lists + initcode
# =========================

function _rowcols_from_idx(idx::Vector{CartesianIndex{2}}, num_rows::Int)
    rowcols = [Int[] for _ in 1:num_rows]
    @inbounds for ij in idx
        push!(rowcols[ij[1]], ij[2])
    end
    return rowcols
end

function initcode(d_nodes::Int, t_nodes::Int, npc::Int; pilot_row_fraction::Float64=0.1)
    code = FEC_create_code(d_nodes, t_nodes, npc)

    idx = findall(!iszero, code.H)
    num_rows, num_cols = size(code.H)
    idrows = _rowcols_from_idx(idx, num_rows)

    idx_colwise = findall(!iszero, code.H')
    cols = _rowcols_from_idx(idx_colwise, num_cols)

    # pilot rows: take last pilot_row_fraction of parity rows (same behavior as your original)
    num_parity_rows = num_rows
    start_row = round(Int, (1.0 - pilot_row_fraction) * num_parity_rows) + 1
    pilot_rows = idrows[start_row:end]
    pilot_indices = sort(unique(vcat(pilot_rows...)))

    return code, cols, idrows, pilot_indices
end

# =========================
# Codeword validity (NO GF(2))
# =========================

# Works for H in sparse CSC and bits in Vector{Int} / BitVector.
function is_valid_codeword(bits::AbstractVector{<:Integer}, H::SparseMatrixCSC{Bool, Int})
    m, _n = size(H)
    s = zeros(Int, m)
    I, J, _ = findnz(H)  # CSC gives all nonzeros
    @inbounds for k in eachindex(I)
        i = I[k]
        j = J[k]
        s[i] ⊻= (bits[j] & 0x1)
    end
    return all(si -> si == 0, s)
end

function is_valid_codeword(H::SparseMatrixCSC{Bool, Int}, x::BitVector)
    bits = Vector{Int}(undef, length(x))
    @inbounds for i in eachindex(x)
        bits[i] = x[i] ? 1 : 0
    end
    return is_valid_codeword(bits, H)
end

# =========================
# Sum-product decoder (SPA)
# =========================
# LLR convention: bit=1 when L_post < 0 (same as your original)

function sum_product_decode(H::SparseMatrixCSC{Bool, Int},
                            y::Vector{Float64},
                            σ²::Float64,
                            parity_indices::Vector{Vector{Int}},
                            col_indices::Vector{Vector{Int}};
                            max_iter::Int=50)

    m, n = size(H)
    length(y) == n || throw(ArgumentError("sum_product_decode: length(y)=$(length(y)) must equal n=$n"))

    # channel LLRs (your original)
    L_ch = @. 2 * y / σ²

    # Messages stored in Dict (keeps it simple; you can optimize later)
    M_vc = Dict{Tuple{Int,Int}, Float64}()  # var j -> check i
    M_cv = Dict{Tuple{Int,Int}, Float64}()  # check i -> var j

    # init var -> check with channel LLR
    @inbounds for j in 1:n
        for i in col_indices[j]
            M_vc[(i,j)] = L_ch[j]
            M_cv[(i,j)] = 0.0
        end
    end

    for iter in 1:max_iter
        # check -> var
        @inbounds for i in 1:m
            neighbors = parity_indices[i]
            d = length(neighbors)
            d == 0 && continue

            tanhs = Vector{Float64}(undef, d)
            for idx in 1:d
                j = neighbors[idx]
                tanhs[idx] = tanh(0.5 * M_vc[(i,j)])
            end

            prefix = ones(Float64, d)
            suffix = ones(Float64, d)

            for idx in 2:d
                prefix[idx] = prefix[idx-1] * tanhs[idx-1]
            end
            for idx in (d-1):-1:1
                suffix[idx] = suffix[idx+1] * tanhs[idx+1]
            end

            for idx in 1:d
                j = neighbors[idx]
                prod_except = prefix[idx] * suffix[idx]
                prod_except = clamp(prod_except, -0.999999, 0.999999)
                M_cv[(i,j)] = 2 * atanh(prod_except)
            end
        end

        # var -> check
        @inbounds for j in 1:n
            neighbors = col_indices[j]
            s_all = L_ch[j]
            for i in neighbors
                s_all += M_cv[(i,j)]
            end
            for i in neighbors
                M_vc[(i,j)] = s_all - M_cv[(i,j)]
            end
        end

        # posterior + early stopping
        L_post = similar(L_ch)
        @inbounds for j in 1:n
            s = L_ch[j]
            for i in col_indices[j]
                s += M_cv[(i,j)]
            end
            L_post[j] = s
        end

        x_hat = @. Int(L_post < 0)
        if is_valid_codeword(x_hat, H)
            return x_hat, iter
        end
    end

    # final posterior
    L_post = similar(L_ch)
    @inbounds for j in 1:n
        s = L_ch[j]
        for i in col_indices[j]
            s += M_cv[(i,j)]
        end
        L_post[j] = s
    end
    return @. Int(L_post < 0), max_iter
end

# =========================
# Exports
# =========================

export Code, FEC_create_code, initcode, encode,
       sum_product_decode, get_H_sparse, is_valid_codeword,
       modulate, demodulate,
       ldpc_paths, load_H_bool

