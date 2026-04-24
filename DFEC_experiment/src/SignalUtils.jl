module SignalUtils
"""
    SignalUtils

Channel estimation and signal processing utilities for DFEC experiments.

Core Functions:
  - estimate_omp_channel: OMP-based channel estimation
  - estimate_mmse_channel: MMSE-based channel estimation
  - argminphase: Phase alignment via BER minimization
  - correct_bpsk_phase_shift: Phase correction for BPSK signals
"""

using Statistics
using SignalAnalysis
using LinearAlgebra
using SparseArrays
using DataFrames

const SampledSignal = SignalAnalysis.SampledSignal

export allpeaks, firstpeak, stdize, estimate_mmse_channel, estimate_sparse_channel,
       estimate_sparse_ls, estimate_omp_channel, do_resample, track_bpsk_carrier_pll,
       unwrap_half_frequency, correct_bpsk_phase_shift, correct_bpsk_packet_phase,
       argminphase

# ============================================================================
# Peak Detection
# ============================================================================
"""
    allpeaks(x::AbstractVector)

Find all peaks in signal above adaptive threshold.
"""
function allpeaks(x::AbstractVector)
    θ = max(maximum(x) / 2.0, 3.5 * median(x))
    peaks = Int[]
    start_index = 1
    while start_index <= length(x)
        next_peak_pos = findfirst(x[start_index:end] .>= θ)
        if next_peak_pos === nothing
            break
        end
        p = start_index + next_peak_pos - 1
        while p < length(x) && x[p+1] > x[p]
            p += 1
        end
        push!(peaks, p)
        start_index = p + 1
    end
    return peaks
end

"""
    firstpeak(x::AbstractVector)

Find first peak in signal above threshold.
"""
function firstpeak(x::AbstractVector)
    θ = max(maximum(x) / 1.5, 3.5 * median(x))
    p = findfirst(x .≥ θ)
    while p !== nothing && p < length(x) && x[p+1] > x[p]
        p += 1
    end
    return p
end

"""
    stdize(y)

Standardize signal to zero mean and unit variance.
"""
function stdize(y)
    y = samples(y)
    ȳ = mean(y)
    μ = y .- ȳ
    σ = sqrt(mean(abs2, μ))
    χ = μ ./ σ
    return χ
end

# ============================================================================
# Channel Estimation Methods
# ============================================================================
"""
    estimate_mmse_channel(y_train::Vector, x_train::Vector, L_h::Int; σ²::Float64=1e-3)

MMSE channel estimation.

# Arguments
  - y_train: Received training sequence
  - x_train: Known training sequence
  - L_h: Channel length
  - σ²: Noise variance (default 1e-3)

# Returns
  Channel impulse response estimate
"""
function estimate_mmse_channel(y_train::Vector, x_train::Vector, L_h::Int; σ²::Float64=1e-3)
    N = length(y_train)
    @assert length(x_train) ≥ N "x_train must be at least as long as y_train"
    X = zeros(ComplexF64, N, L_h)
    for i in 1:N, j in 1:L_h
        if i - j + 1 ≥ 1
            X[i, j] = x_train[i - j + 1]
        end
    end
    XtX = X' * X
    h_est = (XtX + σ² * I(L_h)) \ (X' * y_train)
    return h_est
end

"""
    estimate_sparse_channel(y_train::Vector, x_train::Vector, L_h::Int; σ²::Float64=1e-3, λ::Float64=1e-2)

Sparse channel estimation via proximal gradient descent.
"""
function estimate_sparse_channel(y_train::Vector, x_train::Vector, L_h::Int; σ²::Float64=1e-3, λ::Float64=1e-2)
    N = length(y_train)
    @assert length(x_train) ≥ N "x_train must be at least as long as y_train"
    X = zeros(ComplexF64, N, L_h)
    for i in 1:N, j in 1:L_h
        if i - j + 1 ≥ 1
            X[i, j] = x_train[i - j + 1]
        end
    end
    h = zeros(ComplexF64, L_h)
    α = 1.0 / opnorm(X)^2
    for _ in 1:100
        grad = X' * (X * h - y_train) + σ² * h
        h_temp = h - α * grad
        h = sign.(h_temp) .* max.(abs.(h_temp) .- α * λ, 0.0)
    end
    return h
end

"""
    estimate_sparse_ls(y_train::Vector, x_train::Vector, L_h::Int, k::Int)

k-Sparse LS channel estimation.
"""
function estimate_sparse_ls(y_train::Vector, x_train::Vector, L_h::Int, k::Int)
    N = length(y_train)
    X = zeros(ComplexF64, N, L_h)
    for i in 1:N, j in 1:L_h
        if i - j + 1 ≥ 1
            X[i, j] = x_train[i - j + 1]
        end
    end
    h_ls = X \ y_train
    idx = partialsortperm(abs.(h_ls), rev=true, 1:k)
    h_sparse = zeros(ComplexF64, L_h)
    h_sparse[idx] = h_ls[idx]
    return h_sparse
end

"""
    estimate_omp_channel(y_train::Vector, x_train::Vector, L_h::Int, k::Int)

OMP (Orthogonal Matching Pursuit) channel estimation.

# Arguments
  - y_train: Received training sequence
  - x_train: Known training sequence
  - L_h: Channel length
  - k: Sparsity level

# Returns
  Channel impulse response estimate
"""
function estimate_omp_channel(y_train::Vector, x_train::Vector, L_h::Int, k::Int)
    N = length(y_train)
    X = zeros(ComplexF64, N, L_h)
    for i in 1:N, j in 1:L_h
        if i - j + 1 ≥ 1
            X[i, j] = x_train[i - j + 1]
        end
    end
    residual = copy(y_train)
    support = Int[]
    for _ in 1:k
        correlations = abs.(X' * residual)
        j = argmax(correlations)
        push!(support, j)
        Xs = X[:, support]
        h_tmp = Xs \ y_train
        residual = y_train - Xs * h_tmp
    end
    h_omp = zeros(ComplexF64, L_h)
    h_omp[support] = X[:, support] \ y_train
    return h_omp
end

# ============================================================================
# Resampling and PLL
# ============================================================================
"""
    do_resample(rx_pkt, x_ref, sps::Int, fc::Real, dopp::DataFrame, pbfs::Int, n::Int)

Doppler-tolerant resampling via exhaustive search.
"""
function do_resample(rx_pkt, x_ref, sps::Int, fc::Real, dopp::DataFrame, pbfs::Int, n::Int)
    rx_pb = upconvert(signal(rx_pkt, pbfs / sps), sps, fc)
    ref_pb = upconvert(signal(x_ref, pbfs / sps), sps, fc)
    best_score = -Inf
    best_dop = 0.0
    best_idx = 0
    for dop in -0.98:0.02:1.0
        factor = 1 / (1 + dop)
        if !isfinite(factor)
            continue
        end
        resampled = signal(resample(rx_pb, factor), pbfs)
        cr = mfilter(ref_pb, resampled)
        val, idx = findmax(abs.(cr))
        push!(dopp, (real(val), idx, real(dop)))
        if val > best_score
            best_score, best_dop, best_idx = val, dop, idx
        end
    end
    opt_factor = 1 / (1 + best_dop)
    rx_pb_corr = signal(resample(rx_pb, opt_factor), pbfs)
    rx_bb = downconvert(rx_pb_corr, sps, fc)
    cr = mfilter(x_ref, rx_bb)
    _, align_idx = findmax(abs.(cr))
    if align_idx + n <= length(rx_bb)
        return rx_bb[align_idx : align_idx + n]
    elseif align_idx > 10
        return rx_bb[align_idx - 10 : align_idx - 10 + n]
    else
        return rx_bb[1:n]
    end
end

"""
    track_bpsk_carrier_pll(x, fc, fs, bandwidth=1e-5)

Track and demodulate BPSK signal using PLL.
"""
function track_bpsk_carrier_pll(x, fc, fs, bandwidth=1e-5)
    β = √bandwidth
    ϕ = 0.0
    ω = 0.0
    y = zeros(ComplexF64, length(x))
    demodulated = zeros(length(x))
    phase_errors = zeros(length(x))
    for j in 1:length(x)
        y[j] = cis(-2π * fc * (j-1)/fs + ϕ)
        phase_error = angle(x[j] * conj(y[j]))
        ω += bandwidth * phase_error
        ϕ += β * phase_error + ω
        demodulated[j] = real(x[j] * conj(y[j]))
        phase_errors[j] = phase_error
    end
    return (signal(y, fs), signal(demodulated, fs), signal(phase_errors, fs))
end

"""
    unwrap_half_frequency(signal_2fc)

Extract half-frequency from double-frequency signal.
"""
function unwrap_half_frequency(signal_2fc)
    ph_2f = angle.(signal_2fc)
    ph_2f_unwrap = unwrap(ph_2f)
    ph_fc = ph_2f_unwrap ./ 2
    return cis.(ph_fc)
end

# ============================================================================
# Phase Correction
# ============================================================================
"""
    correct_bpsk_phase_shift(packets, x_datas, Γ, i, spsd, fc, pbfs, n, bpf)

Correct phase shift in BPSK packet using PLL and matched filtering.
"""
function correct_bpsk_phase_shift(packets, x_datas, Γ, i, spsd, fc, pbfs, n, bpf)
    y_data = stdize(packets[i, 1:n])
    yp_pb = upconvert(signal(y_data, pbfs / spsd), spsd, fc)
    y1, _, _ = track_bpsk_carrier_pll(yp_pb, fc, pbfs, Γ)
    ysq_filt = filtfilt(bpf, yp_pb.^2)
    y = pll(ysq_filt, 2fc, 1e-5; fs=pbfs)
    y_half = unwrap_half_frequency(y)
    phase_error = angle.(y1 .* conj(y_half))
    ϕ = unwrap(phase_error)
    y_pb_pll = yp_pb .* exp.(-im .* ϕ)
    y_bb_pll = downconvert(y_pb_pll, spsd, fc)
    ref_data = x_datas[i]
    cr = mfilter(ref_data, y_bb_pll)
    _, ixd = findmax(abs.(cr))
    if ixd + n - 1 < length(y_bb_pll)
        y_data = y_bb_pll[ixd:ixd+n-1]
    elseif ixd > 10
        y_data = y_bb_pll[ixd-10 : ixd-10+n-1]
    else
        y_data = y_bb_pll[1:n]
    end
    return y_data
end

"""
    correct_bpsk_packet_phase(pkt::SampledSignal, x_ref::SampledSignal, Γ::Float64, sps::Int,
                              fc::Float64, pbfs::Float64, n::Int, bpf::Vector, jitter_std::Float64)

Full phase correction pipeline for BPSK packet.
"""
function correct_bpsk_packet_phase(pkt::SampledSignal, x_ref::SampledSignal, Γ::Float64, sps::Int,
                                   fc::Float64, pbfs::Float64, n::Int, bpf::Vector, jitter_std::Float64)
    y_data = stdize(pkt)
    yp_pb = upconvert(y_data, sps, fc)
    y1, _, _ = track_bpsk_carrier_pll(yp_pb, fc, pbfs, Γ)
    ysq_filt = filtfilt(bpf, samples(yp_pb).^2)
    ysq_sig = signal(ysq_filt, framerate(yp_pb))
    pll_bandwidth = jitter_std ≤ π/48 ? 5e-6 : jitter_std ≤ π/90 ? 1e-6 : 1e-5
    y = pll(ysq_sig, 2fc, pll_bandwidth; fs=pbfs)
    y_half = unwrap_half_frequency(y)
    phase_error = angle.(samples(y1) .* conj(samples(y_half)))
    ϕ = unwrap(phase_error)
    corrected_samples = samples(yp_pb) .* exp.(-im .* ϕ)
    y_pb_pll = signal(corrected_samples, framerate(yp_pb))
    y_bb_pll = downconvert(y_pb_pll, sps, fc)
    cr = mfilter(samples(x_ref), samples(y_bb_pll))
    _, ixd = findmax(abs.(cr))
    if ixd + n <= length(y_bb_pll)
        return samples(y_bb_pll)[ixd : ixd + n - 1]
    elseif ixd > 10
        return samples(y_bb_pll)[ixd - 10 : ixd - 10 + n - 1]
    else
        return samples(y_bb_pll)[1:n]
    end
end

# ============================================================================
# Phase Alignment
# ============================================================================
"""
    argminphase(x̂, x)

Find phase rotation that minimizes BER between estimate and reference.

# Arguments
  - x̂: Estimated signal
  - x: Reference signal

# Returns
  (phase_rotated_signal, phase_in_degrees)
"""
function argminphase(x̂, x)
    bers = Float64[]
    range = 0.0:0.1:360.0
    for θ in range
        x̂a = x̂ .* exp(-im * deg2rad(-θ))
        ber = sum(abs, sign.(real.(x̂a)) .!= sign.(real.(x)))
        push!(bers, ber)
    end
    val, id = findmin(bers)
    return (x̂ .* exp(-im * deg2rad(-range[id])), range[id])
end

end # module
