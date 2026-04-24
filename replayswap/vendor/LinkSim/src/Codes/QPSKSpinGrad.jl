# vendor/LinkSim/src/Codes/QPSKSpinGrad.jl
#
# Manual-gradient PSK JSDC core (spin-domain relaxation with pilots).
#
# Supports:
#   - modulation = :qpsk (default)
#   - modulation = :bpsk
#
# Turbo-style soft prior:
#   - L_prior, λ_prior, z_init
#
# Data-term normalization:
#   - σ2_data scales data loss/grad by 1/σ2_data
#
# IMPORTANT parity fix:
#   With mapping bit = (m>0) => m = 2b-1,
#   parity check requires ∏ m_j = (-1)^deg(check).
#   (Not always +1.)
#
# Also uses a stable parity gradient (prefix/suffix products; no division by m).

using Random
using LinearAlgebra

export jsdc_qpsk_manual

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

function psk_from_spins(m::AbstractVector{<:Real}; modulation::Symbol = :qpsk)
    n = length(m)
    if modulation === :qpsk
        @assert iseven(n) "psk_from_spins(:qpsk): length(m) must be even."
        nsym = n ÷ 2
        s = zeros(ComplexF64, nsym)
        inv_sqrt2 = 1 / sqrt(2.0)
        @inbounds for k in 1:nsym
            s[k] = inv_sqrt2 * (m[2k - 1] + 1im * m[2k])
        end
        return s
    elseif modulation === :bpsk
        return ComplexF64.(m)  # imag=0
    else
        error("psk_from_spins: unknown modulation=$modulation (use :qpsk or :bpsk)")
    end
end

function myconv_manual(s::AbstractVector{ComplexF64},
                       h::AbstractVector{ComplexF64})
    Ls = length(s)
    Lh = length(h)
    y = zeros(ComplexF64, Ls + Lh - 1)
    @inbounds for n in 1:length(y)
        acc = 0.0 + 0.0im
        kmin = max(1, n - Lh + 1)
        kmax = min(Ls, n)
        for k in kmin:kmax
            acc += s[k] * h[n - k + 1]
        end
        y[n] = acc
    end
    return y
end

# ------------------------------------------------------------
# Manual-gradient JSDC-PSK decoder
# ------------------------------------------------------------

function jsdc_qpsk_manual(
    y::AbstractVector{ComplexF64},
    code,
    parity_indices::Vector{Vector{Int}},
    pilot_pos::Vector{Int},
    pilot_bpsk::AbstractVector{<:Real},
    h_pos::Vector{Int};
    modulation::Symbol = :qpsk,
    λ_par::Float64 = 0.1,
    λ_pil::Float64 = 1.0,
    γ_z::Float64   = 5e-3,
    γ_h::Float64   = 1e-3,
    η_z::Float64   = 1e-3,
    η_h::Float64   = 1e-2,
    σ2_data::Union{Nothing, Float64} = nothing,
    max_iter::Int  = 200,
    h_init         = nothing,
    L_prior::Union{Nothing, Vector{Float64}} = nothing,
    λ_prior::Float64 = 0.0,
    z_init::Union{Nothing, Vector{Float64}} = nothing,
    verbose::Bool  = false,
)
    n = code.n

    nsym_expected = if modulation === :qpsk
        @assert iseven(n) "jsdc_qpsk_manual(:qpsk): code.n must be even."
        n ÷ 2
    elseif modulation === :bpsk
        n
    else
        error("jsdc_qpsk_manual: unknown modulation=$modulation (use :qpsk or :bpsk)")
    end
    @assert length(y) == nsym_expected "jsdc_qpsk_manual: expected length(y) == $(nsym_expected) for modulation=$(modulation)."
    nsym = length(y)

    # data weight (normalize by noise variance if provided)
    w_data = 1.0
    if σ2_data !== nothing
        w_data = 1.0 / max(Float64(σ2_data), 1e-12)
    end

    # Channel taps (estimated only at provided positions h_pos)
    L = length(h_pos)
    h = if h_init === nothing
        (0.1 / sqrt(2)) .* (randn(L) .+ 1im .* randn(L))
    else
        @assert length(h_init) == L
        ComplexF64.(h_init)
    end

    # Logits & spins
    z = zeros(Float64, n)
    if z_init !== nothing
        @assert length(z_init) == n
        z .= Float64.(z_init)
    end

    pilot_target_spins = Float64.(pilot_bpsk)
    m_checks = length(parity_indices)

    # prior spins from LLR: m0 = tanh(L/2)
    use_prior = (L_prior !== nothing) && (λ_prior > 0)
    m0 = zeros(Float64, n)
    if use_prior
        @assert length(L_prior) == n
        @inbounds for i in 1:n
            m0[i] = tanh(0.5 * L_prior[i])
        end
    end

    # Scratch arrays
    m     = similar(z)
    s     = zeros(ComplexF64, nsym)
    y_hat = similar(y)
    e     = similar(y)

    grad_z        = similar(z)
    grad_h        = similar(h)
    grad_m_data   = similar(z)
    grad_m_parity = similar(z)
    grad_m_pilot  = similar(z)
    grad_m_prior  = similar(z)

    p_c = ones(Float64, m_checks)
    t_c = ones(Float64, m_checks)  # parity target per check: (-1)^deg
    g_s = zeros(ComplexF64, nsym)

    # Parity-gradient scratch (avoid allocations inside loop)
    max_deg = maximum(length.(parity_indices); init=0)
    pref_tmp = ones(Float64, max_deg)
    suff_tmp = ones(Float64, max_deg)

    # Clipping thresholds
    max_norm_h = 5.0
    max_grad_h = 1e3
    max_grad_z = 1e3

    for it in 1:max_iter
        # ---------- forward ----------
        @. m = tanh(z)
        s .= psk_from_spins(m; modulation=modulation)

        y_full = myconv_manual(s, h)
        y_hat .= @view y_full[1:nsym]

        @. e = y_hat - y
        data_loss = w_data * sum(abs2, e)

        # parity term: enforce prod(m[check]) = (-1)^deg(check)
        fill!(p_c, 1.0)
        parity_loss = 0.0
        @inbounds for c in 1:m_checks
            neigh = parity_indices[c]
            d = length(neigh)
            d == 0 && continue
            target = isodd(d) ? -1.0 : 1.0
            t_c[c] = target

            prod_val = 1.0
            for j in neigh
                prod_val *= m[j]
            end
            p_c[c] = prod_val
            δ = (prod_val - target)
            parity_loss += δ * δ
        end

        # pilot term: m[pos] ≈ target (±1)
        pilot_loss = 0.0
        @inbounds for (idx, bit_idx) in enumerate(pilot_pos)
            diff = m[bit_idx] - pilot_target_spins[idx]
            pilot_loss += diff * diff
        end

        # prior term
        prior_loss = 0.0
        if use_prior
            @inbounds for i in 1:n
                d = m[i] - m0[i]
                prior_loss += d*d
            end
        end

        reg_loss = γ_z * sum(abs2, z) + γ_h * sum(abs2, h)
        total_loss = data_loss + λ_par*parity_loss + λ_pil*pilot_loss + λ_prior*prior_loss + reg_loss

        if verbose && (it == 1 || it % 10 == 0 || it == max_iter)
            @info "jsdc_qpsk_manual: iter=$it L=$total_loss " *
                  "Ldata=$data_loss Lpar=$parity_loss Lpil=$pilot_loss Lprior=$prior_loss"
        end
        if !isfinite(total_loss) || !isfinite(data_loss)
            @warn "jsdc_qpsk_manual: non-finite loss at iter=$it, stopping."
            break
        end

        # ---------- backward ----------
        fill!(grad_h, 0.0 + 0.0im)
        fill!(grad_m_data,   0.0)
        fill!(grad_m_parity, 0.0)
        fill!(grad_m_pilot,  0.0)
        fill!(grad_m_prior,  0.0)

        # dL/dh (data + reg)
        @inbounds for ℓ in 1:L
            acc = 0.0 + 0.0im
            for n_idx in 1:nsym
                k = n_idx - ℓ + 1
                if 1 <= k <= nsym
                    acc += e[n_idx] * conj(s[k])
                end
            end
            grad_h[ℓ] = 2 * w_data * acc + 2γ_h * h[ℓ]
        end

        # dL/ds_k (data)
        fill!(g_s, 0.0 + 0.0im)
        @inbounds for k in 1:nsym
            acc = 0.0 + 0.0im
            for n_idx in 1:nsym
                ℓ = n_idx - k + 1
                if 1 <= ℓ <= L
                    acc += e[n_idx] * conj(h[ℓ])
                end
            end
            g_s[k] = 2 * w_data * acc
        end

        # dL_data/dm via symbol mapping
        if modulation === :qpsk
            inv_sqrt2 = 1 / sqrt(2.0)
            @inbounds for k in 1:nsym
                gk = g_s[k]
                iI = 2k - 1
                iQ = 2k
                grad_m_data[iI] += inv_sqrt2 * real(conj(gk))
                grad_m_data[iQ] += inv_sqrt2 * real(1im * conj(gk))
            end
        elseif modulation === :bpsk
            @inbounds for k in 1:nsym
                grad_m_data[k] += real(conj(g_s[k]))
            end
        end

        # dL_par/dm (stable, degree-correct target)
        @inbounds for c in 1:m_checks
            neigh = parity_indices[c]
            d = length(neigh)
            d == 0 && continue
            pc = p_c[c]
            target = t_c[c]
            coeff = 2.0 * (pc - target)   # derivative of (pc-target)^2

            pref_tmp[1] = 1.0
            for t in 2:d
                pref_tmp[t] = pref_tmp[t-1] * m[neigh[t-1]]
            end
            suff_tmp[d] = 1.0
            for t in (d-1):-1:1
                suff_tmp[t] = suff_tmp[t+1] * m[neigh[t+1]]
            end

            for t in 1:d
                j = neigh[t]
                prod_except = pref_tmp[t] * suff_tmp[t]
                grad_m_parity[j] += coeff * prod_except
            end
        end

        # dL_pil/dm
        @inbounds for (idx, bit_idx) in enumerate(pilot_pos)
            grad_m_pilot[bit_idx] += 2.0 * (m[bit_idx] - pilot_target_spins[idx])
        end

        # dL_prior/dm
        if use_prior
            @inbounds for i in 1:n
                grad_m_prior[i] = 2.0 * (m[i] - m0[i])
            end
        end

        # combine -> dL/dz
        @inbounds for i in eachindex(z)
            gmi = grad_m_data[i] +
                  λ_par*grad_m_parity[i] +
                  λ_pil*grad_m_pilot[i] +
                  λ_prior*grad_m_prior[i]
            dmidzi = 1.0 - m[i]^2
            grad_z[i] = gmi * dmidzi + 2γ_z * z[i]
        end

        # clip gradients
        nh = norm(grad_h); nh > max_grad_h && (grad_h .*= (max_grad_h / nh))
        nz = norm(grad_z); nz > max_grad_z && (grad_z .*= (max_grad_z / nz))

        # update
        @. z -= η_z * grad_z
        @. h -= η_h * grad_h

        nh_param = norm(h)
        nh_param > max_norm_h && (h .*= (max_norm_h / nh_param))
    end

    m_final   = tanh.(z)
    xhat_bits = Int.(m_final .> 0.0)

    info = (
        z_final     = z,
        m_final     = m_final,
        h_final     = h,
        modulation  = modulation,
        σ2_data     = (σ2_data === nothing ? NaN : Float64(σ2_data)),
    )

    return xhat_bits, h, info
end
