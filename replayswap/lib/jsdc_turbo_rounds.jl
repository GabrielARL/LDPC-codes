# lib/jsdc_turbo_rounds.jl
#
# JSDC-TURBO rounds with CORRECT LLR polarity for jsdc_qpsk_manual priors.
#
# IMPORTANT:
# - jsdc_qpsk_manual expects L_prior = log P1/P0 (call this L10)
#   because m0 = tanh(L_prior/2) and +1 corresponds to bit=1.
# - Our DFEC/SPA convention uses L01 = log P0/P1 for decoding decisions.
#
# Therefore:
#   L10 = -L01
# and when forming soft means:
#   mean = tanh(L10/2)

# ---- helpers ----
@inline function _llr10_from_m(m::AbstractVector{<:Real})
    mm = Float64.(m)
    @inbounds for i in eachindex(mm)
        mm[i] = clamp(mm[i], -0.999999, 0.999999)
    end
    # L10 = log(P1/P0) for ±1 variable with +1 <-> bit=1
    return 2 .* atanh.(mm)
end

@inline function _damp_vec(old::Vector{Float64}, new::Vector{Float64}, damp::Float64)
    damp <= 0 && return new
    out = similar(new)
    @inbounds for i in eachindex(new)
        out[i] = (1 - damp) * new[i] + damp * old[i]
    end
    return out
end

@inline function _clip_vec!(x::Vector{Float64}, a::Float64)
    a <= 0 && return x
    @inbounds for i in eachindex(x)
        x[i] = clamp(x[i], -a, a)
    end
    return x
end

"""
decode_jsdc_turbo_rounds(...)

Returns:
  (u256_hat, ber_u, ber_in)
"""
function decode_jsdc_turbo_rounds(
    yl::Vector{ComplexF64},
    ul::Vector{Int},
    b512_true::Vector{Int},
    b512_i_true::Vector{Int},
    b1024_true::Vector{Int},
    hl::Vector{ComplexF64},
    codeO, idrowsO, HO::SparseMatrixCSC{Bool,Int}, colsO, parityO,
    codeI, idrowsI, HI::SparseMatrixCSC{Bool,Int}, colsI, parityI,
    itlv::NamedTuple;
    p::Float64,
    jsdc_pil_boost::Float64,
    rounds::Int=2,
    prior_w_in::Float64=0.6,
    prior_w_out::Float64=1.0,
    prior_damp::Float64=0.3,
    prior_clip::Float64=8.0,
    outer_alpha::Float64=1.5,
    # inner hypers
    lampar_in::Float64=1.0,
    lampil_in::Float64=3.0,
    etaz_in::Float64=1e-2,
    gamz_in::Float64=1e-1,
    maxit_in::Int=300,
    # outer hypers
    lampar_out::Float64=1.0,
    lampil_out::Float64=0.5,
    etaz_out::Float64=1.4e-3,
    gamz_out::Float64=1e-3,
    maxit_out::Int=300
)
    # pilots
    pilot_pos_inner = choose_pilots(n2; frac=p)
    pilot_pos_outer = choose_pilots(n1; frac=p)

    pilot_bpsk_in  = bpsk_from_bits(b1024_true[pilot_pos_inner])
    pilot_bpsk_out = bpsk_from_bits(b512_true[pilot_pos_outer])

    # inner channel init
    Lh = min(Lh_in_default, length(hl), length(yl))
    h_pos_in  = collect(1:Lh)
    h_init_in = ComplexF64.(hl[1:Lh])

    # outer memoryless channel
    h_pos_out  = [1]
    h_init_out = ComplexF64[1.0 + 0im]

    # prior states (MUST be L10 = log P1/P0)
    Lprior_in_state  = zeros(Float64, n2)  # 1024 coded bits
    Lprior_out_state = zeros(Float64, n1)  # 512 coded bits
    Lprior_in  = nothing
    Lprior_out = nothing

    info_in_last  = nothing
    info_out_last = nothing

    @assert codeI.icols !== nothing
    inv_ic_I = invperm_vec(collect(codeI.icols))

    for _r in 1:max(1, rounds)
        # ================= INNER JSDC =================
        _xhat_bits_in, _hhat_in, info_in = jsdc_qpsk_manual(
            yl, codeI, parityI,
            pilot_pos_inner, pilot_bpsk_in, h_pos_in;
            λ_par=lampar_in,
            λ_pil=lampil_in * jsdc_pil_boost,
            γ_z=gamz_in, γ_h=3e-4,
            η_z=etaz_in, η_h=3e-4,
            max_iter=maxit_in, h_init=h_init_in,
            L_prior=Lprior_in, λ_prior=prior_w_in,
            z_init=nothing,
            verbose=false
        )
        info_in_last = info_in

        # inner m_final -> L10 coded order
        L1024_10 = _llr10_from_m(info_in.m_final)

        # coded -> perm -> take message -> map to outer order
        L_perm_10   = L1024_10[inv_ic_I]
        L512_msg_10 = Float64.(L_perm_10[end-codeI.k+1:end])                  # b512_i order
        L512_out_10 = itlv.use_itlv ? L512_msg_10[itlv.πinv] : L512_msg_10    # b512 order

        # outer soft observation: mean = tanh(L10/2)
        nsym_out = length(L512_out_10) ÷ 2
        y_out = Vector{ComplexF64}(undef, nsym_out)
        @inbounds for k in 1:nsym_out
            mI = tanh(0.5 * L512_out_10[2k-1])
            mQ = tanh(0.5 * L512_out_10[2k])
            y_out[k] = outer_alpha * ComplexF64(mI, mQ) / sqrt(2)
        end

        # ================= OUTER JSDC =================
        _xhat_bits_out, _hhat_out, info_out = jsdc_qpsk_manual(
            y_out, codeO, parityO,
            pilot_pos_outer, pilot_bpsk_out, h_pos_out;
            λ_par=lampar_out,
            λ_pil=lampil_out * jsdc_pil_boost,
            γ_z=gamz_out, γ_h=1e-3,
            η_z=etaz_out, η_h=1e-2,
            max_iter=maxit_out, h_init=h_init_out,
            L_prior=Lprior_out, λ_prior=prior_w_out,
            z_init=nothing,
            verbose=false
        )
        info_out_last = info_out

        # outer m_final -> L10 in outer coded order (512)
        L512_10_from_outer = _llr10_from_m(info_out.m_final)

        # ===== Exchange priors (all L10) =====
        # outer -> inner message slots
        L_msg_for_inner_10 = itlv.use_itlv ? L512_10_from_outer[itlv.π] : L512_10_from_outer
        L_perm_prior_10 = zeros(Float64, n2)
        L_perm_prior_10[end-codeI.k+1:end] .= L_msg_for_inner_10
        L1024_prior_10 = L_perm_prior_10[codeI.icols]   # perm -> coded

        # inner -> outer prior is the inner-produced outer-order message L10
        L512_prior_for_outer_10 = L512_out_10

        # damp + clip
        Lprior_in_state  = _damp_vec(Lprior_in_state,  L1024_prior_10,          prior_damp)
        Lprior_out_state = _damp_vec(Lprior_out_state, L512_prior_for_outer_10, prior_damp)
        _clip_vec!(Lprior_in_state,  prior_clip)
        _clip_vec!(Lprior_out_state, prior_clip)

        Lprior_in  = Lprior_in_state
        Lprior_out = Lprior_out_state
    end

    # ================= FINAL SPA on outer =================
    # SPA expects L01=log(P0/P1) in DFEC compare_3ways, so use llr_from_m_jsdc (your sign-fixed helper)
    L_spa = llr_from_m_jsdc(Float64.(info_out_last.m_final))
    clip!(L_spa, CLIP_LLR)
    cw_hat_out_spa, _ = sum_product_decode(HO, L_spa, 1.0, idrowsO, colsO; max_iter=MAXITER_SPA)
    u256_hat = Vector{Int}(extract_info_bits(codeO, cw_hat_out_spa))
    ber_u = mean(u256_hat .!= ul)

    # diag: inner coded BER from m_final (bit=1 when m>0)
    ber_in = mean(Int.(Float64.(info_in_last.m_final) .> 0.0) .!= b1024_true)

    # NEW: expose inner soft bits for plotting (already computed, no extra work)
    m_final = Float64.(info_in_last.m_final)

    return (; u256_hat=u256_hat,
              ber_u=Float64(ber_u),
              ber_in=Float64(ber_in),
              m_final=m_final)
end
