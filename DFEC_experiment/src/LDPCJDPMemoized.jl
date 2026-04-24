module LDPCJDPMemoized

using SparseArrays, GaloisFields, Random, LinearAlgebra, Optim, LineSearches, StatsBase, SignalAnalysis
import Optim: minimizer
const GF2 = GaloisField(2)
export Code, sum_product_decode, FEC_create_code, initcode, encode, 
       decode_sparse_joint, myconv, makepacket,
       generate_sparse_channel, modulate, demodulate,
       estimate_channel_from_pilots, resolve_sign_flip,
       is_valid_codeword, prefix_suffix_products
mutable struct Code
    k::Int
    n::Int
    npc::Int
    icols::Union{Nothing, Vector{Int}}
    gen::Union{Nothing, BitMatrix}
    H::Matrix{Bool}
end

Base.show(io::IO, ldpc::Code) = print(io, "LDPC($(ldpc.k)/$(ldpc.n))")

struct JointDecodeResult
    θ::Vector{Float64}
    objective::Float64
    outer_iters::Int
    restart_idx::Int
    valid::Bool
end

minimizer(result::JointDecodeResult) = result.θ

modulate(x; θ=0.0) = x == 1 ? cis(θ) : -cis(θ)
demodulate(x; θ=0.0) = x == 1 ? cis(θ) : -cis(θ)

const H_sparse_cache = IdDict{Code, SparseMatrixCSC{Bool, Int}}()
get_H_sparse(code::Code) = get!(H_sparse_cache, code) do
    sparse(code.H)
end

function readgenerator(filename)
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

function readsparse(filename)
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

function FEC_create_code(k, n, npc)
    Hs = readsparse("$(k)-$(n)-$(npc).H")
    Code(k, n, npc, nothing, nothing, collect(Hs))
end

function encode(ldpc::Code, bits::AbstractVector{<:Integer})
    length(bits) == ldpc.k || throw(ArgumentError("Wrong bit length"))
    if ldpc.icols === nothing
        ldpc.icols, ldpc.gen = readgenerator("$(ldpc.k)-$(ldpc.n)-$(ldpc.npc).gen")
    end
    parity = map(eachrow(ldpc.gen)) do g
        reduce(⊻, g .* bits)
    end |> BitVector
    vcat(parity, bits)[ldpc.icols]
end

function get_row_column_positions(idx::Vector{CartesianIndex{2}}, num_rows::Int)
    rowcols = [Int[] for _ in 1:num_rows]
    for ij in idx
        push!(rowcols[ij[1]], ij[2])
    end
    return rowcols
end

function initcode(d_nodes::Int, t_nodes::Int, npc::Int; pilot_row_fraction::Float64=0.1)
    code = FEC_create_code(d_nodes, t_nodes, npc)
    idx = findall(!iszero, code.H)
    num_rows, num_cols = size(code.H)
    idrows = get_row_column_positions(idx, num_rows)
    idx_colwise = findall(!iszero, code.H')
    cols = get_row_column_positions(idx_colwise, num_cols)
    num_parity_rows = num_rows
    start_row = round(Int, (1.0 - pilot_row_fraction) * num_parity_rows) + 1
    pilot_rows = idrows[start_row:end]
    pilot_indices = sort(unique(vcat(pilot_rows...)))
    return code, cols, idrows, pilot_indices
end

function generate_sparse_channel(L_h::Int, sparsity::Int)
    h = zeros(ComplexF64, L_h)
    pos = sample(1:L_h, sparsity; replace=false)
    h[pos] .= randn(ComplexF64, sparsity)
    return h ./ norm(h)
end

function myconv(x::Vector{<:Number}, h::Vector{<:Number})
    n, L = length(x), length(h)
    [sum(@inbounds h[j] * x[i - j + 1] for j in 1:L if 1 <= i - j + 1 <= n) for i in 1:(n + L - 1)]
end

function linear_conv_grad_x(res::AbstractVector{ComplexF64}, h_pos::AbstractVector{Int},
                            h_vals::AbstractVector{ComplexF64}, n::Int)
    grad = zeros(Float64, n)
    for (tap_idx, tap_pos) in enumerate(h_pos)
        tap_pos > n && continue
        tap = h_vals[tap_idx]
        out_len = n - tap_pos + 1
        @views grad[1:out_len] .+= 2 .* real.(res[tap_pos:n] .* conj(tap))
    end
    return grad
end

function build_supported_conv_matrix(x::AbstractVector{<:Real}, h_pos::AbstractVector{Int}, n::Int)
    A = zeros(ComplexF64, n, length(h_pos))
    for (ii, tap_pos) in enumerate(h_pos)
        tap_pos > n && continue
        out_len = n - tap_pos + 1
        @views A[tap_pos:n, ii] .= ComplexF64.(x[1:out_len])
    end
    return A
end

function parity_loss_and_grad!(g_z::Union{Nothing, Vector{Float64}}, x::Vector{Float64},
                               parity_indices, λ::Float64)
    parity_loss = 0.0
    for inds in parity_indices
        x_inds = x[inds]
        p = prod(x_inds)
        parity_loss += (1 - p)^2
        g_z === nothing && continue

        d = length(inds)
        prefix = ones(Float64, d)
        suffix = ones(Float64, d)
        for jj in 2:d
            prefix[jj] = prefix[jj - 1] * x_inds[jj - 1]
        end
        for jj in (d - 1):-1:1
            suffix[jj] = suffix[jj + 1] * x_inds[jj + 1]
        end
        for (jj, i) in enumerate(inds)
            prod_others = prefix[jj] * suffix[jj]
            g_z[i] += λ * 2 * (1 - p) * (-prod_others) * (1 - x[i]^2)
        end
    end
    return parity_loss
end

function joint_objective(z::Vector{Float64}, y::AbstractVector{ComplexF64}, h_pos::AbstractVector{Int},
                         h_vals::AbstractVector{ComplexF64}, h_prior::AbstractVector{ComplexF64},
                         parity_indices, λ::Float64, γ::Float64, η::Float64)
    n = length(z)
    x = tanh.(z)
    h_full = zeros(ComplexF64, n)
    h_full[h_pos] = h_vals
    ŷ = myconv(x, h_full)
    res = ŷ[1:n] .- y
    parity_loss = parity_loss_and_grad!(nothing, x, parity_indices, λ)
    channel_prior_loss = sum(abs2, h_vals .- h_prior)
    return sum(abs2, res) + λ * parity_loss + γ * (sum(abs2, z) + sum(abs2, h_vals)) + η * channel_prior_loss
end

function solve_supported_channel(x::Vector{Float64}, y::AbstractVector{ComplexF64},
                                 h_pos::AbstractVector{Int}, γ::Float64, η::Float64,
                                 h_prior::AbstractVector{ComplexF64}, n::Int)
    A = build_supported_conv_matrix(x, h_pos, n)
    lhs = A' * A + (γ + η) * I(length(h_pos))
    rhs = A' * ComplexF64.(y) + η .* h_prior
    return lhs \ rhs
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

function estimate_channel_from_pilots(y::Vector{ComplexF64}, pilot_pos::Vector{Int}, pilot_bpsk::Vector{ComplexF64}, L_h::Int)
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

function resolve_sign_flip(x̂::BitVector, z::Vector{Float64}, pilot::Vector{Int}, pilot_bpsk::Vector{ComplexF64}, H::SparseMatrixCSC{Bool, Int})
    vote = sum(real(z[pilot]) .* real(pilot_bpsk))
    x̂_flipped = .!x̂
    H_GF = convert(SparseMatrixCSC{GF2, Int}, H)
    x̂_GF        = GF2.(x̂)
    x̂_flipped_GF = GF2.(x̂_flipped)
    syndrome_raw     = count(!iszero, H_GF * x̂_GF)
    syndrome_flipped = count(!iszero, H_GF * x̂_flipped_GF)
    return (syndrome_flipped < syndrome_raw || vote < 0) ? x̂_flipped : x̂
end

function is_valid_codeword(bits::Vector{Int}, H::SparseMatrixCSC)
    x = GF2.(bits)
    all(H * x .== GF2(0))
end

function is_valid_codeword(H::SparseMatrixCSC{Bool}, x::BitVector)
    is_valid_codeword(collect(Int, x), H)
end

function decode_sparse_joint(y, code::Code, parity_indices, pilot_pos, pilot_bpsk, h_pos;
    λ=1.0, γ=1e-3, η=1.0, h_init=nothing, z_init=nothing,
    max_iter=20, alt_iters=3, num_restarts=1, restart_scales=(1.0,), verbose=false)

    n = length(y)
    y_vec = ComplexF64.(y)
    h_prior = h_init === nothing ? zeros(ComplexF64, length(h_pos)) : ComplexF64.(h_init)
    z_base = z_init === nothing ? zeros(n) : collect(Float64, z_init)
    length(z_base) == n || error("decode_sparse_joint: z_init length $(length(z_base)) != n=$n")

    opt = Optim.Options(f_abstol=1e-3, g_abstol=1e-4, iterations=max_iter)
    scales = collect(Float64.(restart_scales))
    if isempty(scales)
        push!(scales, 1.0)
    end
    while length(scales) < num_restarts
        push!(scales, scales[end])
    end

    best_state = nothing
    H_sparse = get_H_sparse(code)

    for restart_idx in 1:num_restarts
        z_est = scales[restart_idx] .* z_base
        h_vals = h_init === nothing ? randn(ComplexF64, length(h_pos)) : copy(h_prior)
        used_outer_iters = 0

        for outer in 1:alt_iters
            used_outer_iters = outer
            x_est = tanh.(z_est)
            h_vals = solve_supported_channel(x_est, y_vec, h_pos, γ, η, h_prior, n)

            function z_objective(z)
                return joint_objective(z, y_vec, h_pos, h_vals, h_prior, parity_indices, λ, γ, η)
            end

            function z_gradient!(g, z)
                x = tanh.(z)
                h_full = zeros(ComplexF64, n)
                h_full[h_pos] = h_vals
                ŷ = myconv(x, h_full)
                res = ŷ[1:n] .- y_vec
                dLdx = linear_conv_grad_x(res, h_pos, h_vals, n)
                sech2 = 1 .- x.^2
                g .= dLdx .* sech2 .+ 2γ .* z
                parity_loss_and_grad!(g, x, parity_indices, λ)
                return g
            end

            z_result = Optim.optimize(z_objective, z_gradient!, z_est, Optim.LBFGS(), opt)
            z_est = Optim.minimizer(z_result)
        end

        objective = joint_objective(z_est, y_vec, h_pos, h_vals, h_prior, parity_indices, λ, γ, η)
        x̂_raw = tanh.(z_est) .< 0
        x̂ = resolve_sign_flip(x̂_raw, z_est, pilot_pos, pilot_bpsk, H_sparse)
        valid = is_valid_codeword(H_sparse, BitVector(x̂))

        candidate = (
            objective=objective,
            z_est=copy(z_est),
            h_vals=copy(h_vals),
            x_hat=x̂,
            valid=valid,
            outer_iters=used_outer_iters,
            restart_idx=restart_idx,
        )

        if best_state === nothing ||
           (candidate.valid && !best_state.valid) ||
           (candidate.valid == best_state.valid && candidate.objective < best_state.objective)
            best_state = candidate
        end
    end

    θ_opt = vcat(best_state.z_est, real(best_state.h_vals), imag(best_state.h_vals))
    result = JointDecodeResult(
        θ_opt, best_state.objective, best_state.outer_iters, best_state.restart_idx, best_state.valid
    )

    x̂ = best_state.x_hat
    h_norm = norm(best_state.h_vals)
    h_est = h_norm <= eps(Float64) ? best_state.h_vals : best_state.h_vals ./ h_norm

    if verbose
        println("\n📦 Optimization complete.")
        println("✅ Valid codeword: ", result.valid)
        println("🔁 Alternating iterations: ", result.outer_iters)
        println("🔄 Best restart: ", result.restart_idx)
        println("📡 Estimated h: ", h_est)
    end

    return x̂, h_est, result
end

function makepacket(code::Code, num_train::Int, num_data::Int, gap::Int)
    k = code.k
    n = code.n
    packet = Float64[]
    x_train = repeat(mseq(8), num_train)
    append!(packet, x_train)
    packet_gap = fill(0.0, gap)
    append!(packet, packet_gap)
    x_datas = zeros(Float64, num_data, n)
    d_datas = zeros(Float64, num_data, k)
    for i in 1:num_data
       bseq   = mseq(11)[i : k+i-1]
       d_test = Int.((bseq .+ 1) ./ 2)
       E_data = encode(code, d_test)
       x_data = modulate.(E_data)
        x_datas[i, :] = x_data
        d_datas[i, :] = d_test
        append!(packet, x_data)
        if i != num_data
            append!(packet, packet_gap)
        end
    end
    return packet, x_datas, d_datas
end
function sum_product_decode(H::SparseMatrixCSC{Bool}, y::Vector{Float64}, σ²::Float64,
                            parity_indices::Vector{Vector{Int}}, col_indices::Vector{Vector{Int}};
                            max_iter::Int=50)
    m, n = size(H)
    L_ch = @. 2 * y / σ²
    M = Dict{Tuple{Int, Int}, Float64}()

    for j in 1:n
        for i in col_indices[j]
            M[(i, j)] = L_ch[j]
        end
    end

    for iter in 1:max_iter
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

        for j in 1:n
            neighbors = col_indices[j]
            for i in neighbors
                others = setdiff(neighbors, i)
                msg = L_ch[j] + sum((M[(k, j)] for k in others); init=0.0)
                M[(i, j)] = msg
            end
        end

        # Early stopping
        L_post = zeros(Float64, n)
        for j in 1:n
            L_post[j] = L_ch[j] + sum(M[(i, j)] for i in col_indices[j])
        end
        x_hat = @. Int(L_post < 0)
        if is_valid_codeword(x_hat, H)
            return x_hat, iter
        end
    end

    # Final output if no early stopping
    L_post = zeros(Float64, n)
    for j in 1:n
        L_post[j] = L_ch[j] + sum(M[(i, j)] for i in col_indices[j])
    end
    return @. Int(L_post < 0), max_iter
end



end # module
