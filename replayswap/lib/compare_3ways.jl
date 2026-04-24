# lib/compare_3ways.jl
# include-only (no module): utilities for the compare pipeline.
#
# Notes / assumptions:
# - Uses LinkSim trimmed LDPC core where:
#     sum_product_decode(H, y, σ², ...) internally computes L_ch = 2y/σ²
#   So if YOU have LLRs already, use spa_decode_llr() wrapper below.
#
# - EQ+SPA:
#     EQ -> LLR(1024) -> INNER SPA -> hard b512_hat -> (deinterleave) -> OUTER SPA -> u256_hat
#   This is the minimal correct concat flow without requiring SPA to return soft posteriors.

using Random, Statistics, LinearAlgebra, SparseArrays
using JLD2

# Local mapping helpers
include(joinpath(@__DIR__, "ModemQPSK.jl"))
using .ModemQPSK: qpsk_from_bits, bpsk_from_bits

# Load LinkSim (vendored or package)
include(joinpath(@__DIR__, "paths.jl"))
const LS = ensure_linksim_loaded!()

using .LS: initcode, encode, get_H_sparse, sum_product_decode, jsdc_qpsk_manual

# ------------------------------------------------------------
# Constants for your concat setup
# ------------------------------------------------------------
const npc_local = 4
const k1, n1 = 256, 512
const k2, n2 = 512, 1024
const nsym = n2 ÷ 2

const MAXITER_SPA = 80
const CLIP_LLR    = 12.0
const Lh_in_default = 20

# ------------------------------------------------------------
# Dataset loader
# ------------------------------------------------------------
function load_dataset_any(path::String)
    isfile(path) || error("Dataset file not found: $path")
    d = JLD2.load(path)
    if haskey(d, "data") && (d["data"] isa AbstractDict)
        return Dict{String,Any}(d["data"])
    else
        return Dict{String,Any}(d)
    end
end

# ------------------------------------------------------------
# Interleaver from meta_out
# ------------------------------------------------------------
invperm_vec(π::Vector{Int}) = (πinv = similar(π); @inbounds for i in 1:length(π) πinv[π[i]] = i end; πinv)

function get_interleaver(meta_out)::NamedTuple
    use_itlv = false
    π = collect(1:n1)
    πinv = invperm_vec(π)
    if (meta_out isa NamedTuple) && hasproperty(meta_out, :interleaver)
        itlv = getproperty(meta_out, :interleaver)
        if (itlv isa NamedTuple) && hasproperty(itlv, :enabled) && itlv.enabled == true
            use_itlv = true
            if hasproperty(itlv, :π512) && !isempty(itlv.π512)
                π = collect(itlv.π512); πinv = invperm_vec(π)
            elseif hasproperty(itlv, :π512_inv) && !isempty(itlv.π512_inv)
                πinv = collect(itlv.π512_inv); π = invperm_vec(πinv)
            else
                error("meta_out.interleaver.enabled=true but π512/π512_inv missing")
            end
        end
    end
    return (use_itlv=use_itlv, π=π, πinv=πinv)
end

# ------------------------------------------------------------
# Pilots
# ------------------------------------------------------------
function choose_pilots(n::Int; frac::Float64)
    frac <= 0 && return Int[]
    Np = max(1, round(Int, frac * n))
    posf = collect(range(1, stop=n, length=Np))
    return unique!(clamp.(round.(Int, posf), 1, n))
end

function choose_pilots_sym(nsym::Int; frac::Float64)
    frac <= 0 && return Int[]
    Np = max(1, round(Int, frac * nsym))
    posf = collect(range(1, stop=nsym, length=Np))
    return unique!(clamp.(round.(Int, posf), 1, nsym))
end

@inline function clip!(x::Vector{Float64}, a::Float64)
    a <= 0 && return x
    @inbounds for i in eachindex(x)
        xi = x[i]
        x[i] = (xi > a) ? a : ((xi < -a) ? -a : xi)
    end
    return x
end

# ------------------------------------------------------------
# SPA helper: info bits extractor (uses code.icols)
# ------------------------------------------------------------
function extract_info_bits(code, cw::AbstractVector{<:Integer})
    ic = code.icols
    @assert ic !== nothing "code.icols is nothing; call encode(code, ...) once."
    inv = invperm_vec(collect(ic))
    cw_perm = cw[inv]                    # reorder to [parity; bits] systematic layout
    return cw_perm[end-code.k+1:end]     # take last k bits
end

# ------------------------------------------------------------
# Wrapper: run trimmed SPA on *LLRs*
#
# Your trimmed SPA does:
#   L_ch = 2*y/σ²
# So to supply LLRs directly, choose y = 0.5*LLR and σ²=1.
# ------------------------------------------------------------
function spa_decode_llr(H::SparseMatrixCSC{Bool,Int},
                        LLR::Vector{Float64},
                        parity_indices::Vector{Vector{Int}},
                        col_indices::Vector{Vector{Int}};
                        max_iter::Int=MAXITER_SPA)
    y = 0.5 .* LLR
    return sum_product_decode(H, y, 1.0, parity_indices, col_indices; max_iter=max_iter)
end

# ------------------------------------------------------------
# JSDC m_final -> LLR
#
# Keep your previous empirical sign fix:
#   L = -2atanh(m)
# ------------------------------------------------------------
function llr_from_m_jsdc(m::AbstractVector{<:Real})
    mm = Float64.(m)
    @inbounds for i in eachindex(mm)
        mm[i] = clamp(mm[i], -0.999999, 0.999999)
    end
    return -2 .* atanh.(mm)
end

# ------------------------------------------------------------
# LMMSE equalizer used for EQ+SPA
# ------------------------------------------------------------
include(joinpath(@__DIR__, "LMMSE.jl"))
using .LMMSE: lmmse_ff, equalize_soft

# Bit-LLR from equalized QPSK samples, consistent with your mapping:
# bit=1 -> +1, bit=0 -> -1, and LLR is log P(b=0)/P(b=1)
function llr_qpsk_iq_from_z(z::Vector{ComplexF64}, σ2::Float64)
    L = Vector{Float64}(undef, 2length(z))
    s = (2*sqrt(2)) / max(σ2, 1e-12)
    @inbounds for k in 1:length(z)
        L[2k-1] = -s * real(z[k])
        L[2k]   = -s * imag(z[k])
    end
    return L
end

# Hard bits from equalized QPSK symbols (used for DD σ² when p==0)
function hard_bits_from_z_qpsk(z::Vector{ComplexF64})
    b = Vector{Int}(undef, 2length(z))
    @inbounds for k in 1:length(z)
        b[2k-1] = real(z[k]) > 0 ? 1 : 0
        b[2k]   = imag(z[k]) > 0 ? 1 : 0
    end
    return b
end

# ------------------------------------------------------------
# Decode: EQ + SPA (inner then outer)
# ------------------------------------------------------------
function decode_eq_spa(yl::Vector{ComplexF64}, ul::Vector{Int}, b512::Vector{Int}, b1024::Vector{Int}, hl_full::Vector{ComplexF64},
                       codeO, colsO, idrowsO, HO,
                       codeI, colsI, idrowsI, HI,
                       itlv::NamedTuple;
                       p::Float64, M_eq::Int,
                       do_spa::Bool=true)

    pilot_sym_pos = choose_pilots_sym(nsym; frac=p)
    s_true = qpsk_from_bits(b1024)  # used ONLY for pilot rotation and pilot-only σ²

    # equalizer tap window
    Lh = min(Lh_in_default, length(hl_full), nsym)
    h = ComplexF64.(hl_full[1:Lh])

    # initial σ² for LMMSE
    σ2 = 0.5
    wff, _ = lmmse_ff(h, σ2; M=min(M_eq, nsym), D=0)
    z = equalize_soft(yl, wff)

    # size guard
    length(z) < nsym && (z = vcat(z, zeros(ComplexF64, nsym - length(z))))
    length(z) > nsym && (z = z[1:nsym])

    # pilot rotation (puts z onto the QPSK frame) — allowed because uses pilot positions only
    rot = 1.0 + 0im
    if !isempty(pilot_sym_pos)
        c = mean(z[pilot_sym_pos] .* conj.(s_true[pilot_sym_pos]))
        rot = conj(c) / (abs(c) + 1e-12)
        z .*= rot
    end

    # σ² estimate:
    # - if pilots exist: pilot-only residual (allowed)
    # - else: decision-directed residual (NO truth leakage)
    if !isempty(pilot_sym_pos)
        r = z[pilot_sym_pos] .- s_true[pilot_sym_pos]
        σ2 = clamp(mean(abs2, r), 1e-6, 10.0)
    else
        b_hat = hard_bits_from_z_qpsk(z)
        s_hat = qpsk_from_bits(b_hat)
        σ2 = clamp(mean(abs2, z .- s_hat), 1e-6, 10.0)
    end

    # Bit-LLRs from equalized samples (1024)
    L_full = llr_qpsk_iq_from_z(z, σ2)
    clip!(L_full, CLIP_LLR)

    # If you only want "EQ before SPA" for plotting, you can skip SPA
    if !do_spa
        return (; u256_hat=Int[], y_eq=z, L_full=L_full, σ2=σ2, rot=rot)
    end

    # ---------------- INNER SPA ----------------
    xhat_in, _ = spa_decode_llr(HI, L_full, idrowsI, colsI; max_iter=MAXITER_SPA)

    # inner decoded message bits (512) == interleaved outer codeword bits
    b512_hat = extract_info_bits(codeI, xhat_in)

    # convert hard bits to confident LLRs for the outer SPA
    Lhard = CLIP_LLR
    L512_msg = Lhard .* (1 .- 2 .* Float64.(b512_hat))    # b=0 -> +L, b=1 -> -L

    # deinterleave to outer order
    L512_outer = itlv.use_itlv ? L512_msg[itlv.πinv] : L512_msg

    # ---------------- OUTER SPA ----------------
    xhat_out, _ = spa_decode_llr(HO, L512_outer, idrowsO, colsO; max_iter=MAXITER_SPA)
    u256_hat = extract_info_bits(codeO, xhat_out)

    return (; u256_hat=Vector{Int}(u256_hat),
             y_eq=z, L_full=L_full, σ2=σ2, rot=rot,
             b512_hat=Vector{Int}(b512_hat))
end

# ------------------------------------------------------------
# Decode: JSDC (inner) -> map -> JSDC (outer) -> final SPA
# ------------------------------------------------------------
function decode_jsdc_spa(yl::Vector{ComplexF64}, ul::Vector{Int}, b512::Vector{Int}, b1024::Vector{Int}, hl_full::Vector{ComplexF64},
                         codeO, idrowsO, HO, colsO, parityO,
                         codeI, idrowsI, HI, colsI, parityI,
                         itlv::NamedTuple;
                         p::Float64,
                         jsdc_pil_boost::Float64,
                         # ----------------------------
                         # INNER hypers (tunable)
                         # ----------------------------
                         λ_par_in::Float64=1.0,
                         λ_pil_in::Float64=3.0,     # BEFORE boost
                         η_z_in::Float64=1e-2,
                         η_h_in::Float64=3e-4,
                         γ_z_in::Float64=1e-1,
                         γ_h_in::Float64=3e-4,
                         maxit_in::Int=300,
                         # ----------------------------
                         # OUTER hypers (tunable)
                         # ----------------------------
                         alpha_out::Float64=1.0,
                         map_out::Symbol=:tanh,      # :tanh or :linear_clip
                         beta_out::Float64=1.0,      # used only when map_out==:tanh
                         Lscale_out::Float64=6.0,    # used only when map_out==:linear_clip
                         λ_par_out::Float64=0.5,
                         λ_pil_out::Float64=2.0,     # BEFORE boost
                         η_z_out::Float64=3e-4,
                         η_h_out::Float64=1e-2,
                         γ_z_out::Float64=5e-3,
                         γ_h_out::Float64=1e-3,
                         maxit_out::Int=300)

    pilot_pos_inner = choose_pilots(n2; frac=p)
    pilot_pos_outer = choose_pilots(n1; frac=p)

    pilot_bpsk_in  = bpsk_from_bits(b1024[pilot_pos_inner])
    pilot_bpsk_out = bpsk_from_bits(b512[pilot_pos_outer])

    # inner channel init
    Lh = min(Lh_in_default, length(hl_full), length(yl))
    h_pos_in  = collect(1:Lh)
    h_init_in = ComplexF64.(hl_full[1:Lh])

    # apply pilot boost consistently
    λ_pil_in_eff  = λ_pil_in  * jsdc_pil_boost
    λ_pil_out_eff = λ_pil_out * jsdc_pil_boost

    # ---------------- INNER JSDC ----------------
    _xhat_bits_in, _hhat_in, info_in = jsdc_qpsk_manual(
        yl, codeI, parityI,
        pilot_pos_inner, pilot_bpsk_in, h_pos_in;
        λ_par=λ_par_in, λ_pil=λ_pil_in_eff,
        γ_z=γ_z_in, γ_h=γ_h_in,
        η_z=η_z_in, η_h=η_h_in,
        max_iter=maxit_in,
        h_init=h_init_in,
        verbose=false
    )

    # inner m_final -> LLR (SIGN FIX)  (log P0/P1)
    L1024_nat = llr_from_m_jsdc(info_in.m_final)

    # extract message bits LLRs (512) by using codeI.icols permutation
    @assert codeI.icols !== nothing
    inv_ic = invperm_vec(collect(codeI.icols))
    L_perm = L1024_nat[inv_ic]
    L512_msg = Float64.(L_perm[end-codeI.k+1:end])
    L512_outer = itlv.use_itlv ? L512_msg[itlv.πinv] : L512_msg

    # ---------------- OUTER soft "channel" ----------------
    nsym_out = length(L512_outer) ÷ 2
    y_out = Vector{ComplexF64}(undef, nsym_out)

    @inbounds for k in 1:nsym_out
        LI = L512_outer[2k-1]
        LQ = L512_outer[2k]

        mI = 0.0
        mQ = 0.0

        if map_out === :tanh
            mI = tanh(0.5 * beta_out * LI)
            mQ = tanh(0.5 * beta_out * LQ)
        elseif map_out === :linear_clip
            s = max(Lscale_out, 1e-12)
            mI = clamp(LI / s, -1.0, 1.0)
            mQ = clamp(LQ / s, -1.0, 1.0)
        else
            error("decode_jsdc_spa: unknown map_out=$map_out. Use :tanh or :linear_clip.")
        end

        y_out[k] = (alpha_out * ComplexF64(mI, mQ)) / sqrt(2)
    end

    # outer memoryless channel
    h_pos_out  = [1]
    h_init_out = ComplexF64[1.0 + 0im]

    # ---------------- OUTER JSDC ----------------
    _xhat_bits_out, _hhat_out, info_out = jsdc_qpsk_manual(
        y_out, codeO, parityO,
        pilot_pos_outer, pilot_bpsk_out, h_pos_out;
        λ_par=λ_par_out,
        λ_pil=λ_pil_out_eff,
        γ_z=γ_z_out, γ_h=γ_h_out,
        η_z=η_z_out, η_h=η_h_out,
        max_iter=maxit_out,
        h_init=h_init_out,
        verbose=false
    )

    # final SPA from outer soft
    L_spa = llr_from_m_jsdc(info_out.m_final)
    clip!(L_spa, CLIP_LLR)

    cw_hat_out, _ = spa_decode_llr(HO, L_spa, idrowsO, colsO; max_iter=MAXITER_SPA)
    u256_hat = extract_info_bits(codeO, cw_hat_out)

    return (u256_hat=Vector{Int}(u256_hat))
end

export load_dataset_any, get_interleaver,
       choose_pilots, choose_pilots_sym,
       decode_eq_spa, decode_jsdc_spa,
       extract_info_bits, llr_from_m_jsdc
