"""
    run_packets_19_20.jl

Run DFEC decoding on packets 19 and 20 specifically.
Compare results to diagnose the sudden BER change.

Usage:
    julia --project=. scripts/run_packets_19_20.jl
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "config"))

include("../config/experiment_config.jl")
include("../src/SignalUtils.jl")
include("../src/LDPCJDPMemoized.jl")
include("../src/ExperimentCore.jl")

using .ExperimentCore
using .LDPCJDPMemoized
using .SignalUtils
using JLD2, DataFrames, Statistics, Printf
using SignalAnalysis, SignalAnalysis.DSP, Optim
using AdaptiveEstimators

# ============================================================================
# Configuration
# ============================================================================

data_dir = DATA_DIR
if !isabspath(data_dir)
    project_root = dirname(@__DIR__)
    data_dir = joinpath(project_root, data_dir)
end

# ============================================================================
# Load Data
# ============================================================================

println("="^80)
println("🔍 DETAILED ANALYSIS: Packets 19 and 20")
println("="^80 * "\n")

println("📂 Loading data...")
jld2_path = joinpath(data_dir, DATA_FILE)
@load jld2_path all_ytrain_df all_packets_df

y_train_matrix = ExperimentCore.extract_signal_matrix(all_ytrain_df)
packet_matrix = ExperimentCore.extract_signal_matrix(all_packets_df)

x_train = ExperimentCore.prepare_training_signals(NUM_TRAIN)
x_datas = ExperimentCore.prepare_data_signals(
    D_NODES, T_NODES, NPC, NUM_TRAIN, NUM_DATA, GAP, size(y_train_matrix, 1), data_dir
)

tr_len = length(x_train)
n = CODE_N

# Estimate channels
println("🔍 Estimating channels...")
H_est_omp, H_est_mmse = ExperimentCore.estimate_channels(
    y_train_matrix, x_train, H_LEN, K_SPARSE, NOISE_VARIANCE
)

# Create cache structure
cache = ExperimentCore.prepare_cache(
    all_ytrain_df, all_packets_df, x_train, x_datas, H_est_omp, H_est_mmse
)

println("✅ Data loaded and channels estimated\n")

# ============================================================================
# Analysis Function
# ============================================================================

function analyze_packet_detailed(frame_num::Int, cache, pilot_frac::Float64,
                                 d_nodes::Int, t_nodes::Int, npc::Int, n::Int, tr_len::Int,
                                 h_len::Int, λ::Float64, γ::Float64, η::Float64, Q::Vector, data_dir::String)

    # Load code
    old_dir = pwd()
    code = cols = idrows = pilot = nothing
    try
        cd(data_dir)
        code, cols, idrows, pilot = LDPCJDPMemoized.initcode(d_nodes, t_nodes, npc; pilot_row_fraction=pilot_frac)
    finally
        cd(old_dir)
    end

    k, n_code = code.k, code.n
    nonpilot = sort(setdiff(1:n_code, pilot))
    pilot = sort(pilot)
    p_len = length(pilot)

    # Get the correct indices for this packet
    train_idx = cache.packet_to_train_idx[frame_num]
    xdata_idx = cache.packet_to_xdata_idx[frame_num]

    y_train = cache.y_train_matrix[train_idx, 1:length(cache.x_train)]
    y_data = cache.packet_matrix[frame_num, 1:n_code]

    println("\n" * "="^80)
    println("📦 PACKET $frame_num ANALYSIS")
    println("="^80 * "\n")

    # === Step 1: Packet Metrics ===
    println("1️⃣  Packet Metrics:")
    pkt_raw = cache.packet_matrix[frame_num, 1:min(n_code, size(cache.packet_matrix, 2))]
    pkt_power = mean(abs2, pkt_raw)
    pkt_phase_start = angle(pkt_raw[1])
    pkt_phase_end = angle(pkt_raw[end])
    pkt_phase_change = pkt_phase_end - pkt_phase_start

    println("├─ Power: $(round(pkt_power, digits=4))")
    println("├─ Phase at start: $(round(rad2deg(pkt_phase_start), digits=1))°")
    println("├─ Phase at end: $(round(rad2deg(pkt_phase_end), digits=1))°")
    println("└─ Phase progression: $(round(rad2deg(pkt_phase_change), digits=1))°\n")

    # Initialize BER variables (default to error values)
    dfe_ber = 1.0
    spa_ber = 1.0
    grad_ber = 1.0
    x̂ = nothing
    x = nothing

    # === Step 2: DFE Equalization ===
    println("2️⃣  DFE Equalization:")
    try
        r = fit!(DFE(h_len), RLS(),
                [y_train[1:tr_len]; y_data],
                [cache.x_train[1:tr_len]; cache.x_datas[xdata_idx, 1:p_len]],
                tr_len + n_code, nearest(Q))

        x̂ = r.y[tr_len+1:end]
        x = cache.x_datas[xdata_idx, 1:end]
        (_, ph) = SignalUtils.argminphase(x̂, x)
        x̂ = x̂ .* exp(-im * deg2rad(-ph))
        dfe_ber = sum(abs, sign.(real.(x̂[p_len:end])) .!= sign.(real.(x[p_len:end]))) / n_code

        println("├─ DFE phase correction: $(round(ph, digits=1))°")
        println("├─ DFE output power: $(round(mean(abs2, x̂), digits=4))")
        println("└─ **DFE BER: $(round(dfe_ber, digits=4))** ($(round(100*dfe_ber, digits=2))%)\n")
    catch e
        println("❌ DFE failed: $(e)\n")
        return nothing
    end

    # === Step 3: SPA Decoding ===
    println("3️⃣  SPA Decoding:")
    try
        if x̂ !== nothing
            L_ch = 2 .* real.(x̂)
            H_sparse = LDPCJDPMemoized.get_H_sparse(code)
            x_spa, spa_iters = LDPCJDPMemoized.sum_product_decode(H_sparse, real.(x̂), 1.0, idrows, cols)
            x_spa = x_spa[p_len+1:end]
            x_temp = x[p_len+1:end]
            d = LDPCJDPMemoized.demodulate.(x_temp)
            spa_ber = sum(x_spa .!= d) / length(x_spa)

            println("├─ SPA iterations: $spa_iters")
            println("└─ **SPA BER: $(round(spa_ber, digits=4))** ($(round(100*spa_ber, digits=2))%)\n")
        end
    catch e
        println("❌ SPA failed: $(e)\n")
        spa_ber = 1.0
    end

    # === Step 4: Joint Channel & Symbol Decoding ===
    println("4️⃣  Joint Channel & Symbol Decoding:")
    try
        h_full = cache.H_est_omp[train_idx, :]
        h_pos = findall(!iszero, h_full)
        h_init = h_full[h_pos]
        pilot_data = LDPCJDPMemoized.demodulate.(cache.x_datas[xdata_idx, pilot])
        pilot_bpsk = LDPCJDPMemoized.modulate.(pilot_data)

        x̂_bits, h_est, result = LDPCJDPMemoized.decode_sparse_joint(
            cache.packet_matrix[frame_num, 1:n_code], code, idrows, pilot, pilot_bpsk, h_pos;
            h_init=h_init, λ=λ, γ=γ, η=η, verbose=false
        )

        z_opt = Optim.minimizer(result)[1:n_code]
        x̂_soft = tanh.(z_opt)
        x̂3 = x̂_soft[pilot]
        x3 = cache.x_datas[xdata_idx, pilot]
        (_, ph) = SignalUtils.argminphase(x̂3, x3)
        x̂2 = x̂_soft[nonpilot]
        x2 = cache.x_datas[xdata_idx, nonpilot]
        x̂2 = x̂2 .* exp(-im * deg2rad(-ph))
        ber = sum(abs, sign.(real.(x2)) .!= sign.(real.(x̂2))) / n_code

        println("├─ Channel estimate error: $(round(norm(h_est - h_init), digits=4))")
        println("├─ Phase correction: $(round(ph, digits=1))°")
        println("├─ Optimization converged: $(Optim.converged(result))")
        println("└─ **DFEC BER: $(round(ber, digits=4))** ($(round(100*ber, digits=2))%)\n")

        grad_ber = ber
    catch e
        println("❌ Joint decoding failed: $(e)\n")
        grad_ber = 1.0
    end

    # === Summary ===
    println("="^80)
    println("📊 SUMMARY FOR PACKET $frame_num:")
    println("├─ DFE BER:  $(round(dfe_ber, digits=4))")
    println("├─ SPA BER:  $(round(spa_ber, digits=4))")
    println("├─ DFEC BER: $(round(grad_ber, digits=4))")
    println("└─ Best (min): $(round(min(dfe_ber, spa_ber, grad_ber), digits=4))")
    println("="^80 * "\n")

    return (dfe_ber=dfe_ber, spa_ber=spa_ber, grad_ber=grad_ber)
end

# ============================================================================
# Run Packets 19 and 20
# ============================================================================

pilot_frac = 0.36  # Use first pilot fraction

println("\n" * "#"^80)
println("# RUNNING PACKETS 19 AND 20")
println("#"^80 * "\n")

result_19 = analyze_packet_detailed(19, cache, pilot_frac, D_NODES, T_NODES, NPC, n, tr_len,
                                    H_LEN, LAMBDA, GAMMA, ETA, BPSK_CONSTELLATION, data_dir)

result_20 = analyze_packet_detailed(20, cache, pilot_frac, D_NODES, T_NODES, NPC, n, tr_len,
                                    H_LEN, LAMBDA, GAMMA, ETA, BPSK_CONSTELLATION, data_dir)

# ============================================================================
# Comparison
# ============================================================================

println("\n" * "="^80)
println("🔬 COMPARISON: Packet 19 vs Packet 20")
println("="^80 * "\n")

if result_19 !== nothing && result_20 !== nothing
    println("Metric              │ Packet 19  │ Packet 20  │ Change")
    println("─"^60)
    @printf "DFE BER             │ %.4f     │ %.4f     │ %+.4f\n" result_19.dfe_ber result_20.dfe_ber (result_20.dfe_ber - result_19.dfe_ber)
    @printf "SPA BER             │ %.4f     │ %.4f     │ %+.4f\n" result_19.spa_ber result_20.spa_ber (result_20.spa_ber - result_19.spa_ber)
    @printf "DFEC BER            │ %.4f     │ %.4f     │ %+.4f\n" result_19.grad_ber result_20.grad_ber (result_20.grad_ber - result_19.grad_ber)

    println()
    if result_20.dfe_ber > result_19.dfe_ber * 10
        println("🔴 **SIGNIFICANT DEGRADATION** at packet 20")
        println("   BER increased by factor of $(round(result_20.dfe_ber / result_19.dfe_ber, digits=1))")
    elseif result_20.dfe_ber > result_19.dfe_ber
        println("⚠️  Slight degradation at packet 20")
    else
        println("✅ Packet 20 performs better than packet 19")
    end
end

println("\n" * "="^80)
println("💡 INTERPRETATION")
println("="^80 * "\n")

println("""
If packet 20 BER is HIGH while packet 19 is LOW:
├─ Indicates ALIGNMENT/SYNC issue (as diagnosed)
├─ The 28-29 sample shift at packet boundary
├─ DFE trained on wrong symbols in packet 20
└─ Joint decoding cascades the error

If both packets have similar high BER:
├─ Systematic issue affects all packets
├─ Not specific to packet 20
└─ Confirms extraction problem affects entire dataset

If packet 20 is better:
├─ Random variation / lucky alignment
├─ No systematic issue at this boundary
└─ Look at other packet pairs
""")

println("="^80)
