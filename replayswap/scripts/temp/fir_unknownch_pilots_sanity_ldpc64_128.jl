#!/usr/bin/env julia
# scripts/fir_unknownch_pilots_sanity_ldpc64_128.jl
#
# Unknown FIR channel at RX. Use a known BPSK pilot preamble to estimate h (ridge LS),
# estimate σ² from pilot residual, then equalize + decode LDPC(64->128).
#
# Mapping: bit1 -> +1, bit0 -> -1.
#
# Metrics (on cw128 bits):
#   - uncoded hard BER after equalization
#   - SPA BER
#   - JSDC BER (modulation=:bpsk, h init = h_hat, σ² normalized; h frozen by default)
#
# Run:
#   julia --project=. scripts/fir_unknownch_pilots_sanity_ldpc64_128.jl
#   julia --project=. scripts/fir_unknownch_pilots_sanity_ldpc64_128.jl --snr_db 20 --trials 200
#   julia --project=. scripts/fir_unknownch_pilots_sanity_ldpc64_128.jl --h "0.90,0.30,-0.20" --h_len 3 --pilot_len 256
#
# Notes:
# - Pilots are a PREAMBLE (fully known), which avoids the “pilots inside unknown data” circularity.
# - Channel is FIR prefix-conv, output length equals input length (prefix model).

using Random, Printf, Statistics, LinearAlgebra
using SparseArrays

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
const LS = ensure_linksim_loaded!()
using .LS: initcode, encode, get_H_sparse, sum_product_decode, jsdc_qpsk_manual

# ----------------------------
# Helpers
# ----------------------------

# bit1 -> +1, bit0 -> -1
@inline bpsk_map_bit1pos(b::Int) = b == 1 ? 1.0 : -1.0

# Prefix (causal) convolution: y[t] = Σ_{ℓ=1..min(Lh,t)} h[ℓ]*x[t-ℓ+1], for t=1..T
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

# Convolution matrix for prefix model: y ≈ X*h  (X is built from known x)
function Xmat_prefix_from_x(x::Vector{ComplexF64}, Lh::Int)
    T = length(x)
    X = zeros(ComplexF64, T, Lh)
    @inbounds for t in 1:T
        for ℓ in 1:Lh
            idx = t - ℓ + 1
            X[t, ℓ] = (idx >= 1) ? x[idx] : (0.0 + 0im)
        end
    end
    return X
end

# Convolution matrix for prefix model: y ≈ H*x
function convmtx_prefix(h::Vector{ComplexF64}, T::Int)
    Lh = length(h)
    H = zeros(ComplexF64, T, T)
    @inbounds for t in 1:T
        for k in 1:T
            ℓ = t - k + 1
            if 1 <= ℓ <= Lh
                H[t, k] = h[ℓ]
            end
        end
    end
    return H
end

# Genie/LMMSE equalizer for prefix FIR: x̂ = argmin ||y-Hx||^2 + σ2||x||^2
function lmmse_equalize_prefix(y::Vector{ComplexF64}, h::Vector{ComplexF64}, σ2::Float64)
    n = length(y)
    Hc = convmtx_prefix(h, n)
    M = Hc' * Hc
    @inbounds for i in 1:n
        M[i,i] += max(σ2, 1e-12)
    end
    rhs = Hc' * y
    return M \ rhs
end

# Build parity indices list from sparse H: row -> list of variable indices
function build_parity_indices(H::SparseMatrixCSC{Bool, Int})
    m, _n = size(H)
    pi = [Int[] for _ in 1:m]
    I, J, _ = findnz(H)
    @inbounds for (i, j) in zip(I, J)
        push!(pi[i], j)
    end
    return pi
end

@inline ber_bits(a::Vector{Int}, b::Vector{Int}) = mean(a .!= b)

# SPA in your core uses L_ch = 2*y/σ² and decides bit=1 when L_post < 0,
# which corresponds to bit1 -> negative y. Since we use bit1 -> +1 mapping,
# negate soft samples before SPA.
function spa_decode_bits(codeB, colsB, parityB, x_soft::Vector{Float64}, σ2::Float64; max_iter::Int=50)
    H = get_H_sparse(codeB)
    y_for_spa = -x_soft
    xhat, iters = sum_product_decode(H, y_for_spa, max(σ2, 1e-12), parityB, colsB; max_iter=max_iter)
    return Int.(xhat), iters
end

function parse_h_csv(s::String)
    vals = split(replace(s, " " => ""), ",")
    reals = parse.(Float64, vals)
    return ComplexF64.(reals .+ 0im)
end

# Ridge-LS channel estimate from known pilot preamble
function estimate_h_from_pilots(y_pil::Vector{ComplexF64}, x_pil::Vector{ComplexF64}, Lh::Int; rho::Float64=1e-2)
    @assert length(y_pil) == length(x_pil)
    X = Xmat_prefix_from_x(x_pil, Lh)              # Tpil × Lh
    A = X'X
    @inbounds for i in 1:Lh
        A[i,i] += rho
    end
    b = X' * y_pil
    h_hat = A \ b
    y_hat = X * h_hat
    e = y_pil .- y_hat
    # noise variance on complex samples (here imag=0, but keep general)
    σ2_hat = mean(abs2, e) + 1e-12
    return Vector{ComplexF64}(h_hat), Float64(σ2_hat)
end

# ----------------------------
# Main
# ----------------------------

function main()
    snr_db   = 20.0
    trials   = 100
    seed     = 0xC0FFEE
    max_iter_spa = 50

    # True FIR channel (TX uses it; RX does NOT)
    h_true = ComplexF64[0.90 + 0im, 0.30 + 0im, -0.20 + 0im]
    h_len  = length(h_true)
    normalize_h = true

    # Pilot preamble
    pilot_len = 256
    rho_ls = 1e-2

    # JSDC knobs
    jsdc_max_iter = 400
    jsdc_ηz = 3e-4
    jsdc_λpar = 1.0
    jsdc_λprior = 1.0
    jsdc_γz = 5e-3
    jsdc_γh = 1e-3
    jsdc_refine_h = 0   # 0 = freeze h; 1 = allow η_h>0 (small refinement)

    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--snr_db"; i+=1; snr_db = parse(Float64, ARGS[i])
        elseif a == "--trials"; i+=1; trials = parse(Int, ARGS[i])
        elseif a == "--seed"; i+=1; seed = parse(Int, ARGS[i])

        elseif a == "--h"; i+=1; h_true = parse_h_csv(ARGS[i]); h_len = length(h_true)
        elseif a == "--h_len"; i+=1; h_len = parse(Int, ARGS[i])
        elseif a == "--normalize_h"; i+=1; normalize_h = parse(Int, ARGS[i]) != 0

        elseif a == "--pilot_len"; i+=1; pilot_len = parse(Int, ARGS[i])
        elseif a == "--rho_ls"; i+=1; rho_ls = parse(Float64, ARGS[i])

        elseif a == "--jsdc_refine_h"; i+=1; jsdc_refine_h = parse(Int, ARGS[i])
        else
            error("Unknown arg: $a")
        end
        i += 1
    end

    # If user forces h_len different from provided taps, pad/truncate
    if length(h_true) != h_len
        ht = zeros(ComplexF64, h_len)
        L = min(h_len, length(h_true))
        ht[1:L] .= h_true[1:L]
        h_true = ht
    end
    normalize_h && (h_true ./= max(norm(h_true), 1e-12))

    rng = MersenneTwister(seed)

    # LDPC(64->128)
    npc = 4
    codeB, colsB, idrowsB, _ = initcode(64, 128, npc)
    codeB.icols === nothing && (encode(codeB, zeros(Int, 64)); nothing)
    H = get_H_sparse(codeB)
    parityB = build_parity_indices(H)

    n = codeB.n
    k = codeB.k
    @assert n == 128 && k == 64

    # SNR -> noise variance (real AWGN on I)
    # With unit-energy symbols and ||h||^2 = Eh, we set σ2 so output SNR matches snr_db:
    snr_lin = 10.0^(snr_db / 10.0)
    Eh = sum(abs2, h_true)
    σ2 = Eh / (2.0 * snr_lin)
    σ  = sqrt(σ2)

    println("==============================================================")
    @printf("Unknown FIR + pilots | LDPC(%d->%d) BPSK | SNR=%.1f dB | trials=%d\n", k, n, snr_db, trials)
    @printf("Mapping: bit1->+1, bit0->-1 | pilot_len=%d | h_len=%d | rho_ls=%g\n", pilot_len, h_len, rho_ls)
    @printf("normalize_h=%d | ||h_true||^2=%.6f | noise var σ2=%.3e\n", normalize_h ? 1 : 0, Eh, σ2)
    @printf("h_true: [%s]\n", join([@sprintf("%.4f%+.4fj", real(z), imag(z)) for z in h_true], ", "))
    println("==============================================================")

    ber_unc = zeros(Float64, trials)
    ber_spa = zeros(Float64, trials)
    ber_jsd = zeros(Float64, trials)
    spa_it  = zeros(Int, trials)
    mse_h   = zeros(Float64, trials)
    σ2_hatv = zeros(Float64, trials)

    # Fixed pilot preamble bits/symbols (known to RX)
    pil_bits = rand(rng, 0:1, pilot_len)
    pil_sym  = ComplexF64.(Float64[bpsk_map_bit1pos(pil_bits[i]) for i in 1:pilot_len] .+ 0im)

    # JSDC channel positions
    h_pos = collect(1:h_len)

    for t in 1:trials
        # Random info bits and codeword
        u  = rand(rng, 0:1, k)
        cw = Int.(encode(codeB, u))              # 0/1 length 128

        # Map cw bits to BPSK symbols
        xcw_re = Float64[bpsk_map_bit1pos(cw[i]) for i in 1:n]
        xcw    = ComplexF64.(xcw_re .+ 0im)

        # TX sequence = pilot preamble + codeword
        x_tx = vcat(pil_sym, xcw)
        Ttx  = length(x_tx)

        # Channel + AWGN
        y_clean = conv_prefix(h_true, x_tx, Ttx)
        nre = σ .* randn(rng, Ttx)
        y_rx = y_clean .+ ComplexF64.(nre .+ 0im)

        # --- RX channel estimate from pilots ---
        y_pil = @view y_rx[1:pilot_len]
        h_hat, σ2_hat = estimate_h_from_pilots(Vector{ComplexF64}(y_pil), pil_sym, h_len; rho=rho_ls)
        σ2_hatv[t] = σ2_hat
        mse_h[t] = mean(abs2, h_hat .- h_true)

        # --- Equalize CW portion using estimated h and σ² ---
        y_cw = Vector{ComplexF64}(@view y_rx[pilot_len+1 : pilot_len+n])
        x_soft_c = lmmse_equalize_prefix(y_cw, h_hat, σ2_hat)
        x_soft   = real.(x_soft_c)

        # Uncoded hard (after EQ)
        cw_unc = Int.(x_soft .> 0.0)
        ber_unc[t] = ber_bits(cw_unc, cw)

        # SPA (after EQ)
        cw_spa, iters = spa_decode_bits(codeB, colsB, parityB, x_soft, σ2_hat; max_iter=max_iter_spa)
        spa_it[t] = iters
        ber_spa[t] = ber_bits(cw_spa, cw)

        # JSDC (unknown channel, but we use h_hat init; by default freeze h)
        m_init = clamp.(x_soft, -0.999, 0.999)
        z_init = atanh.(m_init)
        L_prior = 2.0 .* z_init

        η_h = (jsdc_refine_h != 0) ? 1e-3 : 0.0

        cw_jsdc, h_jsdc, _info = jsdc_qpsk_manual(
            ComplexF64.(x_soft .* 0 .+ y_cw),  # y_cw already complex; keep explicit
            codeB, parityB,
            Int[], Float64[], h_pos;
            modulation=:bpsk,
            λ_par=jsdc_λpar,
            λ_pil=0.0,
            γ_z=jsdc_γz,
            γ_h=jsdc_γh,
            η_z=jsdc_ηz,
            η_h=η_h,
            max_iter=jsdc_max_iter,
            h_init=h_hat,
            z_init=z_init,
            L_prior=L_prior,
            λ_prior=jsdc_λprior,
            σ2_data=σ2_hat,
            verbose=false
        )
        ber_jsd[t] = ber_bits(Int.(cw_jsdc), cw)

        if t == 1 || t % 20 == 0 || t == trials
            @printf("trial %3d/%d | BER: unc=%.3e spa=%.3e jsdc=%.3e | σ2_hat=%.2e | MSE(h)=%.2e | spa_it=%d\n",
                    t, trials, ber_unc[t], ber_spa[t], ber_jsd[t], σ2_hat, mse_h[t], spa_it[t])
        end
    end

    println("\n---------------- Summary ----------------")
    @printf("Mean BER (cw128): uncoded=%.4e  SPA=%.4e  JSDC=%.4e\n",
            mean(ber_unc), mean(ber_spa), mean(ber_jsd))
    @printf("Median BER       : uncoded=%.4e  SPA=%.4e  JSDC=%.4e\n",
            median(ber_unc), median(ber_spa), median(ber_jsd))
    @printf("h MSE: mean=%.4e  p90=%.4e\n",
            mean(mse_h), sort(mse_h)[clamp(ceil(Int, 0.9*trials), 1, trials)])
    @printf("σ2_hat: mean=%.4e  true=%.4e\n", mean(σ2_hatv), σ2)
    @printf("SPA iters: mean=%.2f  p90=%d\n",
            mean(spa_it), sort(spa_it)[clamp(ceil(Int, 0.9*trials), 1, trials)])
end

main()
