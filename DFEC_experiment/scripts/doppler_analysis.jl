"""
    doppler_analysis.jl

Detect and analyze Doppler shift in received packets.

Doppler effects manifest as:
  - Frequency offset change across packets
  - Phase progression drift
  - Signal alignment degradation after threshold packet
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "config"))

include("../config/experiment_config.jl")
include("../src/SignalUtils.jl")

using JLD2, DataFrames, Plots, Statistics, Printf
using SignalAnalysis

# ============================================================================
# Load Data
# ============================================================================
data_dir = DATA_DIR
if !isabspath(data_dir)
    # @__DIR__ is scripts/, go to parent (DFEC_experiment), then add data_dir
    project_root = dirname(@__DIR__)
    data_dir = joinpath(project_root, data_dir)
end

jld2_path = joinpath(data_dir, DATA_FILE)
@load jld2_path all_ytrain_df all_packets_df

y_train_matrix = Matrix(select(all_ytrain_df, Not([:frame, :block])))
packet_matrix = Matrix(select(all_packets_df, Not([:frame, :block])))

num_frames = size(packet_matrix, 1)
println("📊 Total packets available: $num_frames")

# ============================================================================
# Frequency Offset Estimation
# ============================================================================
"""
    estimate_frequency_offset(y_data::Vector, x_ref::Vector, fs::Real, fc::Real)

Estimate frequency offset using cross-correlation peak shift method.
"""
function estimate_frequency_offset(y_data::Vector, x_ref::Vector, fs::Real, fc::Real)
    # Use PLL to track phase and extract frequency error
    bandwidth = 1e-5
    β = √bandwidth
    ϕ = 0.0
    ω = 0.0
    phase_errors = zeros(length(y_data))

    for j in 1:length(y_data)
        carrier = cis(-2π * fc * (j-1)/fs + ϕ)
        phase_error = angle(y_data[j] * conj(carrier))
        ω += bandwidth * phase_error
        ϕ += β * phase_error + ω
        phase_errors[j] = phase_error
    end

    # Frequency offset from accumulated phase error
    freq_offset = ω / (2π)
    return freq_offset
end

"""
    estimate_phase_drift(y_data::Vector, window_size::Int=100)

Measure phase drift rate across signal.
"""
function estimate_phase_drift(y_data::Vector, window_size::Int=100)
    n = length(y_data)
    if window_size > n
        window_size = n ÷ 2
    end

    phase_drifts = Float64[]
    for start in 1:window_size:(n-window_size)
        seg = y_data[start:(start+window_size-1)]
        phase = angle.(seg)
        # Linear fit to phase evolution
        t = 1:length(phase)
        drift = (phase[end] - phase[1]) / window_size
        push!(phase_drifts, drift)
    end
    return phase_drifts
end

# ============================================================================
# Analysis
# ============================================================================
println("\n" * "="^70)
println("🔍 Doppler Shift Analysis")
println("="^70 * "\n")

# Use training signal as reference
x_train = convert.(ComplexF64, repeat(mseq(8), NUM_TRAIN))
tr_len = length(x_train)
y_ref = convert.(ComplexF64, y_train_matrix[1, 1:tr_len])

# Extract frequency offset from reference training
ref_freq_offset = estimate_frequency_offset(y_ref, x_train, FS, FC)
println("Reference (training) frequency offset: $(round(ref_freq_offset, digits=6)) Hz")

# Analyze frequency offset across packets
freq_offsets = Float64[]
phase_drifts_mean = Float64[]

for frame_num in 1:num_frames
    y_pkt = convert.(ComplexF64, packet_matrix[frame_num, 1:min(50, size(packet_matrix, 2))])

    # Only analyze if we have data
    if length(y_pkt) > 10
        try
            freq_off = estimate_frequency_offset(y_pkt, ones(length(y_pkt)), FS, FC)
            push!(freq_offsets, freq_off)

            drifts = estimate_phase_drift(y_pkt, min(20, length(y_pkt)÷2))
            if !isempty(drifts)
                push!(phase_drifts_mean, mean(drifts))
            end
        catch
            push!(freq_offsets, NaN)
        end
    end
end

# ============================================================================
# Detect Doppler Shift Boundary
# ============================================================================
println("\n📈 Frequency Offset by Packet:")
println("┌────────┬──────────────┬──────────────────┐")
println("│ Packet │ Freq Offset  │ Change from Prev │")
println("├────────┼──────────────┼──────────────────┤")

for i in 1:min(num_frames, 100)
    if !isnan(freq_offsets[i])
        change = i > 1 && !isnan(freq_offsets[i-1]) ? freq_offsets[i] - freq_offsets[i-1] : 0.0
        marker = i == 20 ? " ← Packet 20 threshold" : ""
        @printf "│ %6d │ %12.6f │ %16.6f │%s\n" i freq_offsets[i] change marker
    end
end
println("└────────┴──────────────┴──────────────────┘")

# Analyze before/after packet 20
if length(freq_offsets) >= 20
    before_20 = freq_offsets[1:19]
    after_20 = freq_offsets[21:end]

    before_20_clean = filter(!isnan, before_20)
    after_20_clean = filter(!isnan, after_20)

    if !isempty(before_20_clean) && !isempty(after_20_clean)
        mean_before = mean(before_20_clean)
        mean_after = mean(after_20_clean)
        std_before = std(before_20_clean)
        std_after = std(after_20_clean)
        doppler_shift = mean_after - mean_before

        println("\n🔬 Statistical Comparison:")
        println("├─ Packets 1-19 (Before):")
        println("│  ├─ Mean offset: $(round(mean_before, digits=6)) Hz")
        println("│  └─ Std dev: $(round(std_before, digits=6)) Hz")
        println("├─ Packets 21+ (After):")
        println("│  ├─ Mean offset: $(round(mean_after, digits=6)) Hz")
        println("│  └─ Std dev: $(round(std_after, digits=6)) Hz")
        println("└─ Doppler Shift: $(round(doppler_shift, digits=6)) Hz")

        # Significance test
        if abs(doppler_shift) > std_before + std_after
            println("\n⚠️  SIGNIFICANT DOPPLER SHIFT DETECTED!")
            println("   Shift magnitude: $(round(abs(doppler_shift), digits=3)) Hz")
            println("   This suggests frequency change around packet 20")
        else
            println("\n✅ No significant Doppler shift detected")
            println("   Variation within noise floor")
        end
    end
end

# ============================================================================
# Visualization
# ============================================================================
if !isempty(freq_offsets)
    p = plot(1:length(freq_offsets), freq_offsets, label="Freq Offset",
             linewidth=2, xlabel="Packet Number", ylabel="Frequency Offset (Hz)",
             title="Frequency Offset Across Packets")

    # Add vertical line at packet 20
    vline!([20], label="Packet 20 Threshold", linestyle=:dash, linewidth=2, color=:red)

    # Add before/after mean lines
    if length(freq_offsets) >= 20
        before_mean = mean(filter(!isnan, freq_offsets[1:19]))
        after_mean = mean(filter(!isnan, freq_offsets[21:end]))
        hline!([before_mean], label="Mean (1-19)", linestyle=:dot, alpha=0.7)
        hline!([after_mean], label="Mean (21+)", linestyle=:dot, alpha=0.7)
    end

    png(p, joinpath(dirname(dirname(@__DIR__)), "results", "doppler_analysis.png"))
    println("\n📸 Plot saved to results/doppler_analysis.png")
end

println("\n" * "="^70)
