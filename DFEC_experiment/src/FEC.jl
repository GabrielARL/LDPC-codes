module FEC

using Random
using SparseArrays

export init_systematic_code,
       encode_systematic,
       is_valid,
       bp_decode_spa,
       decode_soft,
       bpsk_hard_bits_on_data,
       bpsk_llrs_on_data,
       qpsk_hard_bits_on_data,
       qpsk_llrs_on_data,
       psk8_hard_bits_on_data,
       psk8_llrs_on_data,
       hard_bits_on_data,
       llrs_on_data

struct SystematicCode
    H::SparseMatrixCSC{Bool, Int}
    parity_sets::Vector{Vector{Int}}
    col_sets::Vector{Vector{Int}}
end

struct BPDecodeResult
    bits::Vector{Int}
    Lpost::Vector{Float64}
    iters::Int
    valid::Bool
end

function _adjacency_lists(H::SparseMatrixCSC{Bool, Int})
    m, n = size(H)
    parity_sets = [findall(!iszero, H[i, :]) for i in 1:m]
    col_sets = [findall(!iszero, H[:, j]) for j in 1:n]
    return parity_sets, col_sets
end

function init_systematic_code(k::Int, m::Int, d_c::Int)
    @assert 1 <= d_c <= m "d_c must be in 1..m"

    Hm = spzeros(Bool, m, k)
    for col in 1:k
        rows = randperm(m)[1:d_c]
        Hm[rows, col] .= true
    end

    Hp = spdiagm(0 => ones(Bool, m))
    H = hcat(Hm, Hp)
    parity_sets, col_sets = _adjacency_lists(H)
    return SystematicCode(H, parity_sets, col_sets)
end

function encode_systematic(H::SparseMatrixCSC{Bool, Int}, msg::AbstractVector{<:Integer})
    m, n = size(H)
    k = n - m
    length(msg) == k || error("encode_systematic: expected $k message bits, got $(length(msg))")

    msg01 = Int.(msg .% 2)
    p_bits = (H[:, 1:k] * msg01) .% 2
    return vcat(msg01, p_bits)
end

function is_valid(H::SparseMatrixCSC{Bool, Int}, bits::AbstractVector{<:Integer})
    length(bits) == size(H, 2) || return false
    syndrome = H * Int.(bits .% 2)
    return all((syndrome .% 2) .== 0)
end

function _edge_check_ids(chk_edges::Vector{Vector{Int}})
    num_edges = isempty(chk_edges) ? 0 : maximum((isempty(edges) ? 0 : maximum(edges)) for edges in chk_edges)
    edge_check = zeros(Int, num_edges)
    for (check_idx, edges) in enumerate(chk_edges)
        for edge in edges
            edge_check[edge] = check_idx
        end
    end
    return edge_check
end

function bp_decode_spa(Lch::Vector{Float64},
                       H::SparseMatrixCSC{Bool, Int},
                       chk_edges::Vector{Vector{Int}},
                       var_edges::Vector{Vector{Int}},
                       edge_var::Vector{Int};
                       maxiter::Int=50,
                       early_stop::Bool=true,
                       clamp_t::Float64=0.999999)

    m, n = size(H)
    length(Lch) == n || error("bp_decode_spa: length(Lch)=$(length(Lch)) != n=$n")

    num_edges = length(edge_var)
    edge_check = _edge_check_ids(chk_edges)
    q = zeros(Float64, num_edges)
    r = zeros(Float64, num_edges)
    Lpost = copy(Lch)
    bits = zeros(Int, n)

    for edge in 1:num_edges
        q[edge] = Lch[edge_var[edge]]
    end

    for iter in 1:maxiter
        for check_idx in 1:m
            edges = chk_edges[check_idx]
            d = length(edges)
            d == 0 && continue

            tanhs = [tanh(0.5 * q[edge]) for edge in edges]
            prefix = ones(Float64, d)
            suffix = ones(Float64, d)

            for idx in 2:d
                prefix[idx] = prefix[idx - 1] * tanhs[idx - 1]
            end
            for idx in (d - 1):-1:1
                suffix[idx] = suffix[idx + 1] * tanhs[idx + 1]
            end
            for idx in 1:d
                prod_except = clamp(prefix[idx] * suffix[idx], -clamp_t, clamp_t)
                r[edges[idx]] = 2 * atanh(prod_except)
            end
        end

        for var_idx in 1:n
            edges = var_edges[var_idx]
            total = Lch[var_idx]
            for edge in edges
                total += r[edge]
            end
            Lpost[var_idx] = total
            bits[var_idx] = Int(total < 0)
            for edge in edges
                q[edge] = total - r[edge]
            end
        end

        if early_stop && is_valid(H, bits)
            return BPDecodeResult(copy(bits), copy(Lpost), iter, true)
        end
    end

    valid = is_valid(H, bits)
    return BPDecodeResult(copy(bits), copy(Lpost), maxiter, valid)
end

function decode_soft(Lch::Vector{Float64},
                     H::SparseMatrixCSC{Bool, Int},
                     chk_edges::Vector{Vector{Int}},
                     var_edges::Vector{Vector{Int}},
                     edge_var::Vector{Int};
                     kwargs...)
    return bp_decode_spa(Lch, H, chk_edges, var_edges, edge_var; kwargs...)
end

function bpsk_hard_bits_on_data(Xhat::Vector{ComplexF64}, data_idx::Vector{Int})
    return Int.(real.(Xhat[data_idx]) .< 0.0)
end

function bpsk_llrs_on_data(Xhat::Vector{ComplexF64},
                           data_idx::Vector{Int},
                           σ2_eff::Float64;
                           clip_llr::Float64=50.0)
    scale = max(σ2_eff, 1e-12)
    return clamp.(2 .* real.(Xhat[data_idx]) ./ scale, -clip_llr, clip_llr)
end

function qpsk_hard_bits_on_data(Xhat::Vector{ComplexF64}, data_idx::Vector{Int})
    bits = Int[]
    for sample in Xhat[data_idx]
        push!(bits, Int(real(sample) < 0.0))
        push!(bits, Int(imag(sample) < 0.0))
    end
    return bits
end

function qpsk_llrs_on_data(Xhat::Vector{ComplexF64},
                           data_idx::Vector{Int},
                           σ2_eff::Float64;
                           clip_llr::Float64=50.0)
    scale = max(σ2_eff, 1e-12)
    llrs = Float64[]
    for sample in Xhat[data_idx]
        push!(llrs, clamp(2 * real(sample) / scale, -clip_llr, clip_llr))
        push!(llrs, clamp(2 * imag(sample) / scale, -clip_llr, clip_llr))
    end
    return llrs
end

function psk8_hard_bits_on_data(Xhat::Vector{ComplexF64}, data_idx::Vector{Int})
    error("psk8_hard_bits_on_data is not implemented in this minimal FEC module")
end

function psk8_llrs_on_data(Xhat::Vector{ComplexF64},
                           data_idx::Vector{Int},
                           σ2_eff::Float64;
                           clip_llr::Float64=50.0)
    error("psk8_llrs_on_data is not implemented in this minimal FEC module")
end

function _canonical_modulation(mod::Symbol)
    mod_upper = Symbol(uppercase(String(mod)))
    if mod_upper in (:BPSK, :QPSK, :PSK8, Symbol("8PSK"))
        return mod_upper == Symbol("8PSK") ? :PSK8 : mod_upper
    end
    error("Unsupported modulation: $(mod)")
end

function hard_bits_on_data(Xhat::Vector{ComplexF64},
                           data_idx::Vector{Int};
                           mod::Symbol=:QPSK)
    modc = _canonical_modulation(mod)
    if modc === :BPSK
        return bpsk_hard_bits_on_data(Xhat, data_idx)
    elseif modc === :QPSK
        return qpsk_hard_bits_on_data(Xhat, data_idx)
    end
    return psk8_hard_bits_on_data(Xhat, data_idx)
end

function llrs_on_data(Xhat::Vector{ComplexF64},
                      data_idx::Vector{Int},
                      σ2_eff::Float64;
                      mod::Symbol=:QPSK,
                      clip_llr::Float64=50.0)
    modc = _canonical_modulation(mod)
    if modc === :BPSK
        return bpsk_llrs_on_data(Xhat, data_idx, σ2_eff; clip_llr=clip_llr)
    elseif modc === :QPSK
        return qpsk_llrs_on_data(Xhat, data_idx, σ2_eff; clip_llr=clip_llr)
    end
    return psk8_llrs_on_data(Xhat, data_idx, σ2_eff; clip_llr=clip_llr)
end

end # module FEC
