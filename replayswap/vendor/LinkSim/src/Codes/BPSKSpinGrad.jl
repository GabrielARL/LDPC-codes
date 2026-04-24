# vendor/LinkSim/src/Codes/BPSKSpinGrad.jl
#
# Manual-gradient BPSK JSDC core (spin-domain relaxation with pilots).
#
# Robustness upgrades vs the older QPSKSpinGrad approach:
#   (1) Loss/grad normalization (data/parity/pilot/prior) so λ are stable
#   (2) Parity products use clamped m_par to avoid vanishing parity gradients near m≈0
#   (3) Simple step decay η_eff = η/(1+η_decay*(it-1))
#   (4) Optional freeze_h_after to prevent channel drift killing good basins
#
# Signature is BPSK-only:
#   y length must equal code.n (one symbol per bit)
#   pilot_pos are bit indices in 1..n
#   pilot_bpsk are ±1 targets (same length as pilot_pos)
#
# Returns:
#   xhat_bits :: Vector{Int} (0/1)
#   h_final   :: Vector{ComplexF64}
#   info      :: NamedTuple (z_final, m_final, losses, etc.)
#
using Random
using LinearAlgebra

export jsdc_bpsk_manual

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
@inline bpsk_from_spins(m::AbstractVector{<:Real}) = ComplexF64.(m)  # imag=0

function myconv_manual(x::AbstractVector{ComplexF64}, h::AbstractVector{ComplexF64})
    Lx = length(x)
    Lh = length(h)
    y = zeros(ComplexF64, Lx + Lh - 1)
    @inbounds for n in 1:length(y)
        acc = 0.0 + 0.0im
        kmin = max(1, n - Lh + 1)
        kmax = min(Lx, n)
        for k in kmin:kmax
            acc += x[k] * h[n - k + 1]
        end
        y[n] = acc
    end
    return y
end

# ------------------------------------------------------------
# Robust BPSK JSDC
# ------------------------------------------------------------
function jsdc_bpsk_manual(
    y::AbstractVector{ComplexF64},
    code,
    parity_indices::Vector{Vector{Int}},
    pilot_pos::Vector{Int},
    pilot_bpsk::AbstractVector{<:Real},
    h_pos::Vector{Int};
    # weights
    λ_par::Float64 = 1.0,
    λ_pil::Float64 = 20.0,
    γ_z::Float64   = 5e-3,
    γ_h::Float64   = 1e-3,
    # step sizes
    η_z::Float64   = 1e-3,
    η_h::Float64   = 1e-2,
    η_decay::Float64 = 1e-3,
    freeze_h_after::Int = 50,
    # noise scaling
    σ2_data::Union{Nothing, Float64} = nothing,
    max_iter::Int  = 300,
    # init
    h_init = nothing,
    z_init::Union{Nothing, Vector{Float64}} = nothing,
    # turbo-style prior
    L_prior::Union{Nothing, Vector{Float64}} = nothing,
    λ_prior::Float64 = 1.0,
    # robustness / clipping
    normalize_terms::Bool = true,
    eps_m_par::Float64 = 1e-3,
    max_norm_h::Float64 = 5.0,
    max_grad_h::Float64 = 1e3,
    max_grad_z::Float64 = 1e3,
    verbose::Bool = false,
)
    n = code.n
    @assert length(y) == n "jsdc_bpsk_manual: expected length(y)==code.n ($n). Got $(length(y))."

    # data weight (normalize by noise variance if provided)
    w_data = 1.0
    if σ2_data !== nothing
        w_data = 1.0 / max(Float64(σ2_data), 1e-12)
    end

    # Channel taps estimated at positions h_pos only
    L = length(h_pos)
    h = if h_init === nothing
        (0.1 / sqrt(2)) .* (randn(L) .+ 1im .* randn(L))
    else
        @assert length(h_init) == L
        ComplexF64.(h_init)
    end

    # logits
    z = zeros(Float64, n)
    if z_init !== nothing
        @assert length(z_init) == n
        z .= Float64.(z_init)
    end

    m_checks = length(parity_indices)
    pilot_target = Float64.(pilot_bpsk)
    use_prior = (L_prior !== nothing) && (λ_prior > 0)
    m0 = zeros(Float64, n)
    if use_prior
        @assert length(L_prior) == n
        @inbounds for i in 1:n
            m0[i] = tanh(0.5 * L_prior[i])
        end
    end

    # normalization scalars
    inv_n     = normalize_terms ? (1.0 / max(n, 1)) : 1.0
    inv_chk   = normalize_terms ? (1.0 / max(m_checks, 1)) : 1.0
    inv_pil   = normalize_terms ? (1.0 / max(length(pilot_pos), 1)) : 1.0
    inv_L     = 1.0 / max(L, 1)

    # scratch
    m      = similar(z)
    m_par  = similar(z)
    x      = zeros(ComplexF64, n)
    y_hat  = similar(y)
    e      = similar(y)

    grad_h        = similar(h)
    grad_z        = similar(z)
    grad_m_data   = similar(z)
    grad_m_parity = similar(z)
    grad_m_pilot  = similar(z)
    grad_m_prior  = similar(z)

    p_c = ones(Float64, m_checks)
    t_c = ones(Float64, m_checks)

    max_deg = maximum(length.(parity_indices); init=0)
    pref_tmp = ones(Float64, max_deg)
    suff_tmp = ones(Float64, max_deg)

    last_losses = (data=NaN, par=NaN, pil=NaN, prior=NaN, reg=NaN, total=NaN)

    for it in 1:max_iter
        # ---------- forward ----------
        @. m = tanh(z)

        # clamp m only for parity products (avoid vanishing parity gradients)
        if eps_m_par > 0
            @inbounds for i in 1:n
                mi = m[i]
                if abs(mi) < eps_m_par
                    m_par[i] = (mi >= 0) ? eps_m_par : -eps_m_par
                else
                    m_par[i] = mi
                end
            end
        else
            m_par .= m
        end

        x .= bpsk_from_spins(m)
        y_full = myconv_manual(x, h)
        y_hat .= @view y_full[1:n]
        @. e = y_hat - y

        data_loss = w_data * sum(abs2, e) * inv_n

        # parity: prod(m) = (-1)^deg
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
                prod_val *= m_par[j]
            end
            p_c[c] = prod_val
            δ = prod_val - target
            parity_loss += δ*δ
        end
        parity_loss *= inv_chk

        # pilots
        pilot_loss = 0.0
        @inbounds for (k, idx) in enumerate(pilot_pos)
            d = m[idx] - pilot_target[k]
            pilot_loss += d*d
        end
        pilot_loss *= inv_pil

        # prior
        prior_loss = 0.0
        if use_prior
            @inbounds for i in 1:n
                d = m[i] - m0[i]
                prior_loss += d*d
            end
            prior_loss *= inv_n
        end

        reg_loss = (γ_z * sum(abs2, z) * inv_n) + (γ_h * sum(abs2, h) * inv_L)
        total_loss = data_loss + λ_par*parity_loss + λ_pil*pilot_loss + λ_prior*prior_loss + reg_loss
        last_losses = (data=data_loss, par=parity_loss, pil=pilot_loss, prior=prior_loss, reg=reg_loss, total=total_loss)

        if verbose && (it == 1 || it % 10 == 0 || it == max_iter)
            @info "jsdc_bpsk_manual iter=$it L=$total_loss data=$data_loss par=$parity_loss pil=$pilot_loss prior=$prior_loss"
        end
        if !isfinite(total_loss) || !isfinite(data_loss)
            @warn "jsdc_bpsk_manual: non-finite loss at iter=$it, stopping."
            break
        end

        # ---------- backward ----------
        fill!(grad_h, 0.0 + 0.0im)
        fill!(grad_m_data,   0.0)
        fill!(grad_m_parity, 0.0)
        fill!(grad_m_pilot,  0.0)
        fill!(grad_m_prior,  0.0)

        # dL/dh
        @inbounds for ℓ in 1:L
            acc = 0.0 + 0.0im
            for n_idx in 1:n
                k = n_idx - ℓ + 1
                if 1 <= k <= n
                    acc += e[n_idx] * conj(x[k])
                end
            end
            grad_h[ℓ] = 2*w_data*inv_n*acc + 2*(γ_h*inv_L)*h[ℓ]
        end

        # dL/dx_k (data), then dL/dm_k since x=m for BPSK
        @inbounds for k in 1:n
            acc = 0.0 + 0.0im
            for n_idx in 1:n
                ℓ = n_idx - k + 1
                if 1 <= ℓ <= L
                    acc += e[n_idx] * conj(h[ℓ])
                end
            end
            # x_k is real (m_k), so take real(conj(...))
            grad_m_data[k] += real(conj(2*w_data*inv_n*acc))
        end

        # parity gradient (prefix/suffix; uses m_par products)
        @inbounds for c in 1:m_checks
            neigh = parity_indices[c]
            d = length(neigh)
            d == 0 && continue
            pc = p_c[c]
            target = t_c[c]
            coeff = 2.0 * inv_chk * (pc - target)

            pref_tmp[1] = 1.0
            for t in 2:d
                pref_tmp[t] = pref_tmp[t-1] * m_par[neigh[t-1]]
            end
            suff_tmp[d] = 1.0
            for t in (d-1):-1:1
                suff_tmp[t] = suff_tmp[t+1] * m_par[neigh[t+1]]
            end

            for t in 1:d
                j = neigh[t]
                prod_except = pref_tmp[t] * suff_tmp[t]
                grad_m_parity[j] += coeff * prod_except
            end
        end

        # pilot gradient
        @inbounds for (k, idx) in enumerate(pilot_pos)
            grad_m_pilot[idx] += 2.0 * inv_pil * (m[idx] - pilot_target[k])
        end

        # prior gradient
        if use_prior
            @inbounds for i in 1:n
                grad_m_prior[i] = 2.0 * inv_n * (m[i] - m0[i])
            end
        end

        # combine -> dL/dz
        @inbounds for i in 1:n
            gmi = grad_m_data[i] +
                  λ_par*grad_m_parity[i] +
                  λ_pil*grad_m_pilot[i] +
                  λ_prior*grad_m_prior[i]
            dmidzi = 1.0 - m[i]^2
            grad_z[i] = gmi * dmidzi + 2.0*(γ_z*inv_n)*z[i]
        end

        # clip gradients
        nh = norm(grad_h); nh > max_grad_h && (grad_h .*= (max_grad_h / nh))
        nz = norm(grad_z); nz > max_grad_z && (grad_z .*= (max_grad_z / nz))

        # step schedule
        ηz_eff = η_z / (1.0 + η_decay*(it-1))
        ηh_eff = η_h / (1.0 + η_decay*(it-1))

        # update
        @. z -= ηz_eff * grad_z
        if freeze_h_after <= 0 || it <= freeze_h_after
            @. h -= ηh_eff * grad_h
        end

        # h norm cap
        nhp = norm(h)
        nhp > max_norm_h && (h .*= (max_norm_h / nhp))
    end

    m_final = tanh.(z)
    xhat_bits = Int.(m_final .> 0.0)

    info = (
        z_final = z,
        m_final = m_final,
        h_final = h,
        σ2_data = (σ2_data === nothing ? NaN : Float64(σ2_data)),
        losses  = last_losses,
        normalize_terms = normalize_terms,
        eps_m_par = eps_m_par,
        η_decay = η_decay,
        freeze_h_after = freeze_h_after,
    )
    return xhat_bits, h, info
end
