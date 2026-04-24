module LDPCJDPMemoized

using SparseArrays, GaloisFields, Random, LinearAlgebra, Optim, StatsBase, SignalAnalysis

const GF2 = GaloisField(2)

export Code,
       get_H_sparse,
       sum_product_decode,
       FEC_create_code,
       initcode,
       encode,
       decode_sparse_joint,
       myconv,
       makepacket,
       generate_sparse_channel,
       modulate,
       demodulate,
       estimate_channel_from_pilots,
       resolve_sign_flip,
       is_valid_codeword,
       prefix_suffix_products

# ----------------------------
# Types
# ----------------------------
mutable struct Code
    k::Int
    n::Int
    npc::Int
    icols::Union{Nothing, Vector{Int}}
    gen::Union{Nothing, BitMatrix}
    H::Matrix{Bool}
end

Base.show(io::IO, ldpc::Code) = print(io, "LDPC($(ldpc.k)/$(ldpc.n))")

# ----------------------------
# Paths (anchor to repo)
# If this file is lib/LDPCJDPMemoized.jl, LDPC files are at ../codes/ldpc/
# ----------------------------
const DEFAULT_LDPC_DIR = normpath(joinpath(@__DIR__, "..", "codes", "ldpc"))

ldpc_path(k::Int, n::Int, npc::Int, ext::String; base_dir::String=DEFAULT_LDPC_DIR) =
    joinpath(base_dir, "$(k)-$(n)-$(npc).$ext")

# ----------------------------
# Modem (BPSK)
# Convention: bit1 -> +1, bit0 -> -1
# modulate: bit -> complex ±1
# demodulate: complex -> bit 0/1 (1 if real>=0)
# ----------------------------
modulate(b::Integer; θ::Real=0.0) = (b == 1 ? cis(θ) : -cis(θ))

"Demodulate received symbol -> bit (1 if real(x) >= 0 else 0)."
demodulate(x::Number; θ::Real=0.0) = (real(x * cis(-θ)) >= 0 ? 1 : 0)

# ----------------------------
# H sparse cache
# ----------------------------
const H_sparse_cache = IdDict{Code, SparseMatrixCSC{Bool, Int}}()
get_H_sparse(code::Code) = get!(H_sparse_cache, code) do
    sparse(code.H)
end

# ----------------------------
# Read LDPC files
# ----------------------------
function readgenerator(filename::String)
    open(filename) do io
        read(io, UInt32) == 0x00004780 || error("Bad generator")
        read(io, UInt8) == 0x64 || error("Bad generator: must be dense")
        p = Int(read(io, UInt32))
        n = Int(read(io, UInt32))
        icols = [Int(read(io, UInt32)) + 1 for _ in 1:n] |> invperm
        Int(read(io, UInt32)) == p || error("Bad row size")
        Int(read(io, UInt32)) == n - p || error("Bad column size")
        G = mapreduce(hcat, 1:n - p) do _
            v = [read(io, UInt32) for _ in 1:ceil(Int, p / 32)]
            isodd(length(v)) && push!(v, 0)
            b = BitArray(undef, p)
            b.chunks .= reinterpret(UInt64, v)
            b
        end
        icols, G
    end
end

function readsparse(filename::String)
    ii, jj = Int[], Int[]
    for s in readlines(filename)
        m = match(r"^ *(\d+):(.+)$", s)
        if m !== nothing
            i = parse(Int, m[1])
            js = parse.(Int, split(m[2]))
            append!(ii, repeat([i + 1], length(js)))
            append!(jj, js .+ 1)
        end
    end
    sparse(ii, jj, true)
end

function FEC_create_code(k::Int, n::Int, npc::Int; base_dir::String=DEFAULT_LDPC_DIR)
    Hs = readsparse(ldpc_path(k, n, npc, "H"; base_dir=base_dir))
    Code(k, n, npc, nothing, nothing, collect(Hs))
end

function encode(ldpc::Code, bits::AbstractVector{<:Integer}; base_dir::String=DEFAULT_LDPC_DIR)
    length(bits) == ldpc.k || throw(ArgumentError("Wrong bit length"))
    if ldpc.icols === nothing
        ldpc.icols, ldpc.gen = readgenerator(ldpc_path(ldpc.k, ldpc.n, ldpc.npc, "gen"; base_dir=base_dir))
    end
    parity = map(eachrow(ldpc.gen)) do g
        reduce(⊻, g .* bits)
    end |> BitVector
    vcat(parity, bits)[ldpc.icols]
end

# ----------------------------
# Index helpers
# ----------------------------
function get_row_column_positions(idx::Vector{CartesianIndex{2}}, num_rows::Int)
    rowcols = [Int[] for _ in 1:num_rows]
    for ij in idx
        push!(rowcols[ij[1]], ij[2])
    end
    return rowcols
end

function initcode(d_nodes::Int, t_nodes::Int, npc::Int;
                  pilot_row_fraction::Float64=0.1,
                  base_dir::String=DEFAULT_LDPC_DIR)
    code = FEC_create_code(d_nodes, t_nodes, npc; base_dir=base_dir)
    idx = findall(!iszero, code.H)
    num_rows, num_cols = size(code.H)
    idrows = get_row_column_positions(idx, num_rows)

    idx_colwise = findall(!iszero, code.H')
    cols = get_row_column_positions(idx_colwise, num_cols)

    # choose pilot columns from last pilot_row_fraction of parity rows
    start_row = round(Int, (1.0 - pilot_row_fraction) * num_rows) + 1
    pilot_rows = idrows[start_row:end]
    pilot_indices = sort(unique(vcat(pilot_rows...)))
    return code, cols, idrows, pilot_indices
end

# ----------------------------
# Channel utilities
# ----------------------------
function generate_sparse_channel(L_h::Int, sparsity::Int)
    h = zeros(ComplexF64, L_h)
    pos = sample(1:L_h, sparsity; replace=false)
    h[pos] .= randn(ComplexF64, sparsity)
    return h ./ max(norm(h), 1e-12)
end

# ----------------------------
# Convolution
# ----------------------------
function myconv(x::Vector{<:Number}, h::Vector{<:Number})
    n, L = length(x), length(h)
    [sum(@inbounds h[j] * x[i - j + 1] for j in 1:L if 1 <= i - j + 1 <= n) for i in 1:(n + L - 1)]
end

function prefix_suffix_products(x::Vector{Float64})
    n = length(x)
    prefix = ones(Float64, n)
    suffix = ones(Float64, n)
    for i in 2:n
        @inbounds prefix[i] = prefix[i - 1] * x[i - 1]
    end
    for i in (n - 1):-1:1
        @inbounds suffix[i] = suffix[i + 1] * x[i + 1]
    end
    return prefix, suffix
end

function estimate_channel_from_pilots(y::Vector{ComplexF64}, pilot_pos::Vector{Int},
                                      pilot_bpsk::Vector{ComplexF64}, L_h::Int)
    X = zeros(ComplexF64, length(pilot_pos), L_h)
    for (i, p) in enumerate(pilot_pos)
        for j in 1:L_h
            if p - j + 1 ∈ 1:length(pilot_bpsk)
                @inbounds X[i, j] = pilot_bpsk[p - j + 1]
            end
        end
    end
    return X \ y[pilot_pos]
end

# ----------------------------
# Global flip resolve (pilot vote + syndrome)
# ----------------------------
function resolve_sign_flip(xhat::BitVector, z::Vector{Float64},
                           pilot::Vector{Int}, pilot_bpsk::Vector{ComplexF64},
                           H::SparseMatrixCSC{Bool, Int})
    vote = isempty(pilot) ? 0.0 : sum(real(z[pilot]) .* real(pilot_bpsk))
    xhat_flipped = .!xhat

    H_GF = convert(SparseMatrixCSC{GF2, Int}, H)
    xhat_GF         = GF2.(xhat)
    xhat_flipped_GF = GF2.(xhat_flipped)

    syndrome_raw     = count(!iszero, H_GF * xhat_GF)
    syndrome_flipped = count(!iszero, H_GF * xhat_flipped_GF)

    return (syndrome_flipped < syndrome_raw || vote < 0) ? xhat_flipped : xhat
end

# ----------------------------
# Codeword validity
# ----------------------------
function is_valid_codeword(bits::Vector{Int}, H::SparseMatrixCSC)
    x = GF2.(bits)
    all(H * x .== GF2(0))
end

function is_valid_codeword(H::SparseMatrixCSC{Bool, Int}, x::BitVector)
    is_valid_codeword(collect(Int, x), H)
end

# ----------------------------
# Joint sparse optimizer decoder (NOW uses pilots via λ_pil)
# ----------------------------
function decode_sparse_joint(y, code::Code, parity_indices, pilot_pos, pilot_bpsk, h_pos;
    λ::Float64=1.0,
    λ_pil::Float64=0.0,   # <-- NEW
    γ::Float64=1e-3,
    η::Float64=1.0,
    h_init=nothing,
    max_iter::Int=20,
    verbose::Bool=false)

    n = length(y)
    h_prior = h_init === nothing ? randn(ComplexF64, length(h_pos)) : h_init
    θ0 = vcat(zeros(n), real(h_prior), imag(h_prior))

    function loss_and_grad!(g, θ)
        z   = @view θ[1:n]
        h_r = @view θ[n+1:n+length(h_pos)]
        h_i = @view θ[n+length(h_pos)+1:end]
        h_vals = ComplexF64.(h_r, h_i)

        x = tanh.(z)

        h_full = zeros(ComplexF64, n)
        h_full[h_pos] = h_vals
        ŷ = myconv(x, h_full)
        res = @view(ŷ[1:n]) .- y

        # data term (approx grad as in your original)
        dLdx = 2 .* real.(res .* conj.(@view h_full[1:n]))
        sech2 = 1 .- x.^2
        g[1:n] .= dLdx .* sech2 .+ 2γ .* z

        # channel grads (approx as in your original)
        channel_shifts = Dict{Int, Vector{Float64}}()
        for (ii, j) in enumerate(h_pos)
            shifted = get!(channel_shifts, j, circshift(x, j - 1))
            g[n + ii] = 2 * real(sum(res .* conj.(shifted))) + 2γ * h_r[ii]
            g[n + length(h_pos) + ii] = 2 * imag(sum(res .* conj.(shifted))) + 2γ * h_i[ii]
        end

        # parity term
        parity_loss = 0.0
        for inds in parity_indices
            p = prod(x[inds])
            parity_loss += (1 - p)^2
            for i in inds
                g[i] += λ * 2 * (1 - p) * (-p / x[i]) * (1 - x[i]^2)
            end
        end

        # --- NEW: pilot term (forces x[pilot] ≈ ±1) ---
        pil_loss = 0.0
        if !isempty(pilot_pos) && λ_pil > 0
            @inbounds for (kk, j) in enumerate(pilot_pos)
                t = (real(pilot_bpsk[kk]) >= 0) ? 1.0 : -1.0
                e = x[j] - t
                pil_loss += e * e
                g[j] += λ_pil * 2 * e * (1 - x[j]^2)
            end
        end

        return sum(abs2, res) + λ * parity_loss + λ_pil * pil_loss +
               γ * (sum(abs2, z) + sum(abs2, h_r) + sum(abs2, h_i))
    end

    opt = Optim.Options(f_abstol=1e-3, g_abstol=1e-4, iterations=max_iter)
    result = Optim.optimize(x -> loss_and_grad!(zeros(length(θ0)), x), θ0, Optim.LBFGS(), opt)

    θ_opt = Optim.minimizer(result)
    z_opt = θ_opt[1:n]
    h_est = ComplexF64.(θ_opt[n+1:n+length(h_pos)], θ_opt[n+length(h_pos)+1:end])

    x̂_raw = BitVector(tanh.(z_opt) .< 0)
    x̂ = resolve_sign_flip(x̂_raw, z_opt, pilot_pos, pilot_bpsk, get_H_sparse(code))

    return x̂, h_est ./ max(norm(h_est), 1e-12), result
end


# ----------------------------
# Packet builder
# (training + LDPC codewords driven by mseq8/mseq11 sequences you pass in)
# mseq8/mseq11 should be Int vectors of ±1
# ----------------------------
function makepacket(code::Code,
                    num_train::Int,
                    num_data::Int,
                    gap::Int,
                    mseq8::Vector{Int},
                    mseq11::Vector{Int};
                    base_dir::String=DEFAULT_LDPC_DIR)
    k = code.k
    n = code.n

    packet = Float64[]

    x_train = Float64.(repeat(mseq8, num_train))
    append!(packet, x_train)
    append!(packet, fill(0.0, gap))

    x_datas = zeros(Float64, num_data, n)
    d_datas = zeros(Float64, num_data, k)

    for i in 1:num_data
        bseq   = mseq11[i : k+i-1]              # ±1
        d_test = Int.((bseq .+ 1) ./ 2)         # -> 0/1
        cw     = encode(code, d_test; base_dir=base_dir)
        bits   = Int.(cw)

        # store as Float64 ±1 for your older pipeline
        x_data = Float64.(2 .* bits .- 1)
        x_datas[i, :] = x_data
        d_datas[i, :] = d_test

        append!(packet, x_data)
        if i != num_data
            append!(packet, fill(0.0, gap))
        end
    end

    return packet, x_datas, d_datas
end

# ----------------------------
# SPA (sum-product) decoder
# ----------------------------
function sum_product_decode(H::SparseMatrixCSC{Bool, Int},
                            y::Vector{Float64}, σ²::Float64,
                            parity_indices::Vector{Vector{Int}},
                            col_indices::Vector{Vector{Int}};
                            max_iter::Int=50)
    m, n = size(H)
    L_ch = @. 2 * y / σ²
    M = Dict{Tuple{Int, Int}, Float64}()

    for j in 1:n, i in col_indices[j]
        M[(i, j)] = L_ch[j]
    end

    for iter in 1:max_iter
        # check node update
        for i in 1:m
            neighbors = parity_indices[i]
            d = length(neighbors)
            tanhs = [tanh(0.5 * M[(i, j)]) for j in neighbors]
            prefix = ones(Float64, d)
            suffix = ones(Float64, d)
            for j in 2:d
                prefix[j] = prefix[j-1] * tanhs[j-1]
            end
            for j in (d-1):-1:1
                suffix[j] = suffix[j+1] * tanhs[j+1]
            end
            for j in 1:d
                prod_except = prefix[j] * suffix[j]
                M[(neighbors[j], i)] = 2 * atanh(clamp(prod_except, -0.999999, 0.999999))
            end
        end

        # variable node update
        for j in 1:n
            neighbors = col_indices[j]
            for i in neighbors
                msg = L_ch[j] + sum((M[(k, j)] for k in neighbors if k != i); init=0.0)
                M[(i, j)] = msg
            end
        end

        # posterior & early stop
        L_post = zeros(Float64, n)
        for j in 1:n
            L_post[j] = L_ch[j] + sum(M[(i, j)] for i in col_indices[j])
        end
        x_hat = @. Int(L_post < 0)
        if is_valid_codeword(x_hat, H)
            return x_hat, iter
        end
    end

    L_post = zeros(Float64, n)
    for j in 1:n
        L_post[j] = L_ch[j] + sum(M[(i, j)] for i in col_indices[j])
    end
    return @. Int(L_post < 0), max_iter
end

end # module
