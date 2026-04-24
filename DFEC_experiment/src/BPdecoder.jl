# ============================================================
# src/Algorithms/BPdecoder.jl
# Compatibility shim replacing old quarantine/BPdecoder.jl
# using the new src/Phy/FEC stack.
# ============================================================
module BPdecoder

using SparseArrays
using ..FEC

export initcode,
       encode_ldpc_systematic,
       sum_product_decode,
       prprp_decode,
       is_valid_codeword

# ------------------------------------------------------------
# Old API: initcode(k, d_c)
# Builds a rate-1/2 systematic code and returns:
#   H, parity_indices, col_indices
# ------------------------------------------------------------
function initcode(k::Int, d_c::Int)
    m = k  # keep old rate-1/2 convention
    code = FEC.init_systematic_code(k, m, d_c)
    H = code.H
    parity_indices = code.parity_sets
    col_indices    = code.col_sets
    return H, parity_indices, col_indices
end

# ------------------------------------------------------------
# Old API wrapper
# old: encode_ldpc_systematic(H, msg::Vector{Bool})
# new: FEC.encode_systematic(H, msg01)
# ------------------------------------------------------------
function encode_ldpc_systematic(H::SparseMatrixCSC{Bool,Int}, msg::Vector{Bool})
    return FEC.encode_systematic(H, Int.(msg))
end

function encode_ldpc_systematic(H::SparseMatrixCSC{Bool,Int}, msg::AbstractVector{<:Integer})
    return FEC.encode_systematic(H, Int.(msg))
end

# ------------------------------------------------------------
# Old API wrapper
# old: is_valid_codeword(bits, H)
# new: FEC.is_valid(H, bits)
# ------------------------------------------------------------
function is_valid_codeword(bits::AbstractVector{<:Integer}, H::SparseMatrixCSC{Bool,Int})
    return FEC.is_valid(H, Int.(bits))
end

# ------------------------------------------------------------
# Build edge lists needed by new BP decoder from old parity lists
# ------------------------------------------------------------
function _edge_lists_from_parity(parity_indices::Vector{Vector{Int}}, n::Int)
    m = length(parity_indices)
    chk_edges = [Int[] for _ in 1:m]
    var_edges = [Int[] for _ in 1:n]
    edge_var  = Int[]

    eid = 0
    for i in 1:m
        for j in parity_indices[i]
            eid += 1
            push!(chk_edges[i], eid)
            push!(var_edges[j], eid)
            push!(edge_var, j)
        end
    end

    return chk_edges, var_edges, edge_var
end

# ------------------------------------------------------------
# Old API:
#   sum_product_decode(H, y, σ², parity_indices, col_indices; max_iter=50)
#
# old y was BPSK observation, converted internally to LLR = 2y/σ²
# returns hard bits only
# ------------------------------------------------------------
function sum_product_decode(H::SparseMatrixCSC{Bool,Int},
                            y::Vector{Float64},
                            σ²::Float64,
                            parity_indices::Vector{Vector{Int}},
                            col_indices::Vector{Vector{Int}};
                            max_iter::Int=50)

    m, n = size(H)
    length(y) == n || error("sum_product_decode: length(y)=$(length(y)) != n=$n")

    # old convention
    Lch = @. 2.0 * y / max(σ², 1e-12)

    chk_edges, var_edges, edge_var = _edge_lists_from_parity(parity_indices, n)

    dec = FEC.bp_decode_spa(
        Lch, H, chk_edges, var_edges, edge_var;
        maxiter=max_iter,
        early_stop=false
    )

    return dec.bits
end

# ------------------------------------------------------------
# Old API:
#   prprp_decode(H, y, σ², parity_indices, col_indices; ...)
#
# Returns:
#   (x_hat=dblk, bitpr=bprb, iters=it, valid=true/false)
#
# We emulate this using the new SPA decoder.
# ------------------------------------------------------------
function prprp_decode(H::SparseMatrixCSC{Bool,Int},
                      y::Vector{Float64},
                      σ²::Float64,
                      parity_indices::Vector{Vector{Int}},
                      col_indices::Vector{Vector{Int}};
                      max_iter::Int=50,
                      stop_on_valid::Bool=true,
                      eps_t::Float64=1e-12)

    m, n = size(H)
    length(y) == n || error("prprp_decode: length(y)=$(length(y)) != n=$n")

    # old convention:
    # L_ch = 2y/σ²
    Lch = @. 2.0 * y / max(σ², 1e-12)

    chk_edges, var_edges, edge_var = _edge_lists_from_parity(parity_indices, n)

    dec = FEC.bp_decode_spa(
        Lch, H, chk_edges, var_edges, edge_var;
        maxiter=abs(max_iter),
        early_stop=stop_on_valid,
        clamp_t=1.0 - max(eps_t, 1e-12)
    )

    # old code exposed probability bit=1
    # with LLR convention: P(bit=1) = 1 / (1 + exp(Lpost))
    bitpr = @. 1.0 / (1.0 + exp(dec.Lpost))

    return (
        x_hat = dec.bits,
        bitpr = bitpr,
        Lpost = dec.Lpost,
        iters = dec.iters,
        valid = dec.valid
    )
end

end # module
