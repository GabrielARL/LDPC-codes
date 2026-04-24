"""
    correct_phase_drift_and_save_jld2.jl

Phase-correct the saved DFEC JLD2 dataset and write out a new JLD2 file with the
same object names:

  - `all_ytrain_df`
  - `all_packets_df`

Method:
  1. Estimate channels from the saved training rows
  2. Fit a linear phase ramp to each training row against its predicted received signal
  3. Re-estimate channels from the corrected training rows
  4. Fit a linear phase ramp to each packet row against its predicted received signal
  5. Save corrected DataFrames with the original schema

Usage:

    julia --project=. scripts/correct_phase_drift_and_save_jld2.jl

Optional output path:

    julia --project=. scripts/correct_phase_drift_and_save_jld2.jl /tmp/corrected.jld2
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "config"))

include("../config/experiment_config.jl")
include("../src/SignalUtils.jl")
include("../src/LDPCJDPMemoized.jl")
include("../src/ExperimentCore.jl")

using .SignalUtils
using .LDPCJDPMemoized
using .ExperimentCore

using JLD2, DataFrames, Printf, Statistics, LinearAlgebra

const DEFAULT_OUTPUT_FILE = "logged_packets_and_ytrain_phase_corrected.jld2"
const EPS = 1e-12

function resolve_data_dir()
    data_dir = DATA_DIR
    if !isabspath(data_dir)
        project_root = dirname(@__DIR__)
        data_dir = joinpath(project_root, data_dir)
    end
    return data_dir
end

function resolve_output_path(data_dir::String)
    if isempty(ARGS)
        return joinpath(data_dir, DEFAULT_OUTPUT_FILE)
    end

    out = ARGS[1]
    return isabspath(out) ? out : joinpath(pwd(), out)
end

function unwrap_phase(ph::AbstractVector{<:Real})
    out = collect(Float64, ph)
    for i in 2:length(out)
        δ = out[i] - out[i - 1]
        if δ > π
            out[i:end] .-= 2π
        elseif δ < -π
            out[i:end] .+= 2π
        end
    end
    return out
end

function weighted_line_fit(t::Vector{Float64}, y::Vector{Float64}, w::Vector{Float64})
    sw = sum(w)
    sw <= EPS && return (0.0, 0.0)

    st = sum(w .* t)
    sy = sum(w .* y)
    stt = sum(w .* t .* t)
    sty = sum(w .* t .* y)
    denom = sw * stt - st^2

    if abs(denom) <= EPS
        intercept = sy / sw
        return (intercept, 0.0)
    end

    slope = (sw * sty - st * sy) / denom
    intercept = (sy - slope * st) / sw
    return (intercept, slope)
end

function fit_phase_ramp(y::AbstractVector{ComplexF64}, y_ref::AbstractVector{ComplexF64})
    n = min(length(y), length(y_ref))
    n > 0 || error("Cannot fit phase ramp on empty vectors")

    yv = y[1:n]
    rv = y_ref[1:n]
    z = yv .* conj.(rv)
    phase_err = unwrap_phase(angle.(z))
    t = collect(0.0:(n - 1))
    w = max.(abs.(rv).^2, EPS)

    intercept, slope = weighted_line_fit(t, phase_err, w)
    phase_model = intercept .+ slope .* t
    y_corr = yv .* exp.(-im .* phase_model)
    score = abs(sum(y_corr .* conj.(rv))) / (norm(y_corr) * norm(rv) + EPS)
    residual = angle.(y_corr .* conj.(rv))
    residual_std = std(residual)

    return (
        intercept = intercept,
        slope = slope,
        score = score,
        residual_std = residual_std
    )
end

function apply_phase_ramp(y::AbstractVector{ComplexF64}, intercept::Float64, slope::Float64, start_idx::Int)
    positions = collect(1:length(y))
    phase_model = intercept .+ slope .* (positions .- start_idx)
    return y .* exp.(-im .* phase_model)
end

function predicted_received_signal(x_ref::AbstractVector{<:Number}, h::AbstractVector{ComplexF64}, n::Int)
    y_ref = LDPCJDPMemoized.myconv(ComplexF64.(x_ref), h)
    return y_ref[1:n]
end

function replace_signal_columns!(df::DataFrame, corrected_matrix)
    signal_cols = names(select(df, Not(ExperimentCore.metadata_columns(df))))
    length(signal_cols) == size(corrected_matrix, 2) ||
        error("Signal column count $(length(signal_cols)) does not match corrected matrix width $(size(corrected_matrix, 2))")

    for (col_idx, col_name) in enumerate(signal_cols)
        df[!, col_name] = corrected_matrix[:, col_idx]
    end
    return df
end

function correct_training_rows(y_train_matrix, x_train, H_est)
    corrected = similar(ComplexF64.(y_train_matrix))
    slopes = Float64[]
    scores = Float64[]

    for row in 1:size(y_train_matrix, 1)
        h = H_est[row, :]
        y_ref = predicted_received_signal(x_train, h, size(y_train_matrix, 2))
        fit = fit_phase_ramp(ComplexF64.(y_train_matrix[row, :]), y_ref)
        corrected[row, :] = apply_phase_ramp(
            ComplexF64.(y_train_matrix[row, :]), fit.intercept, fit.slope, 1
        )
        push!(slopes, fit.slope)
        push!(scores, fit.score)
    end

    return corrected, slopes, scores
end

function choose_packet_fit(packet_row, x_ref, h)
    y_full = ComplexF64.(packet_row)
    y_ref = predicted_received_signal(x_ref, h, CODE_N)
    candidates = NamedTuple[]

    for start_idx in 1:(length(y_full) - CODE_N + 1)
        fit = fit_phase_ramp(y_full[start_idx:start_idx + CODE_N - 1], y_ref)
        push!(candidates, merge((start_idx = start_idx,), fit))
    end

    best = candidates[1]
    for cand in candidates[2:end]
        if cand.score > best.score + 1e-9
            best = cand
        elseif abs(cand.score - best.score) <= 1e-9 && cand.residual_std < best.residual_std
            best = cand
        end
    end

    return best
end

function correct_packet_rows(cache, packet_matrix, H_est)
    corrected = similar(ComplexF64.(packet_matrix))
    slopes = Float64[]
    scores = Float64[]
    chosen_windows = Int[]

    for row in 1:size(packet_matrix, 1)
        train_idx = cache.packet_to_train_idx[row]
        xdata_idx = cache.packet_to_xdata_idx[row]
        h = H_est[train_idx, :]
        fit = choose_packet_fit(packet_matrix[row, :], cache.x_datas[xdata_idx, :], h)
        corrected[row, :] = apply_phase_ramp(
            ComplexF64.(packet_matrix[row, :]), fit.intercept, fit.slope, fit.start_idx
        )
        push!(slopes, fit.slope)
        push!(scores, fit.score)
        push!(chosen_windows, fit.start_idx)
    end

    return corrected, slopes, scores, chosen_windows
end

function print_summary(label, slopes, scores)
    println(label)
    println("├─ Rows: $(length(slopes))")
    println("├─ Mean |slope|: $(round(mean(abs.(slopes)), digits=6)) rad/sample")
    println("├─ Median |slope|: $(round(median(abs.(slopes)), digits=6)) rad/sample")
    println("├─ Mean fit score: $(round(mean(scores), digits=4))")
    println("└─ Median fit score: $(round(median(scores), digits=4))\n")
end

data_dir = resolve_data_dir()
output_path = resolve_output_path(data_dir)
mkpath(dirname(output_path))

println("="^88)
println("PHASE-DRIFT CORRECTION FOR SAVED DFEC JLD2")
println("="^88 * "\n")
println("Input data dir: $(data_dir)")
println("Output file:    $(output_path)\n")

jld2_path = joinpath(data_dir, DATA_FILE)
@load jld2_path all_ytrain_df all_packets_df

y_train_matrix = ExperimentCore.extract_signal_matrix(all_ytrain_df)
packet_matrix = ExperimentCore.extract_signal_matrix(all_packets_df)

x_train = ExperimentCore.prepare_training_signals(NUM_TRAIN)
x_datas = ExperimentCore.prepare_data_signals(
    D_NODES, T_NODES, NPC, NUM_TRAIN, NUM_DATA, GAP, nrow(all_ytrain_df), data_dir
)

println("Estimating channels from original training rows...")
H_est_omp, H_est_mmse = ExperimentCore.estimate_channels(
    y_train_matrix, x_train, H_LEN, K_SPARSE, NOISE_VARIANCE
)

println("Correcting training rows...")
corrected_y_train_matrix, train_slopes, train_scores = correct_training_rows(
    y_train_matrix, x_train, H_est_omp
)
print_summary("Training-row correction summary:", train_slopes, train_scores)

println("Re-estimating channels from corrected training rows...")
H_est_omp_corr, H_est_mmse_corr = ExperimentCore.estimate_channels(
    corrected_y_train_matrix, x_train, H_LEN, K_SPARSE, NOISE_VARIANCE
)

corrected_all_ytrain_df = deepcopy(all_ytrain_df)
replace_signal_columns!(corrected_all_ytrain_df, corrected_y_train_matrix)

cache_for_packets = ExperimentCore.prepare_cache(
    corrected_all_ytrain_df, all_packets_df, x_train, x_datas, H_est_omp_corr, H_est_mmse_corr
)

println("Correcting packet rows...")
corrected_packet_matrix, packet_slopes, packet_scores, chosen_windows = correct_packet_rows(
    cache_for_packets, packet_matrix, H_est_omp_corr
)
print_summary("Packet correction summary:", packet_slopes, packet_scores)
println("Packet-window choices:")
println("├─ Window `s1:s128` chosen: $(count(==(1), chosen_windows))")
println("└─ Window `s2:s129` chosen: $(count(==(2), chosen_windows))\n")

corrected_all_packets_df = deepcopy(all_packets_df)
replace_signal_columns!(corrected_all_packets_df, corrected_packet_matrix)

all_ytrain_df = corrected_all_ytrain_df
all_packets_df = corrected_all_packets_df

@save output_path all_ytrain_df all_packets_df

println("✅ Saved corrected dataset")
println("   Path: $(output_path)")
println("   Objects: all_ytrain_df, all_packets_df")
