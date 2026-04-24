module ExperimentCore
"""
    ExperimentCore

Core experiment orchestration for DFEC (Decision Feedback Equalizer + LDPC decoding).

This module ties together channel estimation, DFE, and joint LDPC decoding
to evaluate system performance over a range of pilot symbol fractions.

Main Pipeline:
  1. Load signal and packet data
  2. Estimate channels using OMP and MMSE
  3. For each pilot fraction:
     - Initialize LDPC code
     - For each frame:
       - Apply DFE to received signal
       - Run SPA decoding
       - Run joint channel/symbol decoding
       - Compute BER metrics
  4. Save results to CSV
"""

using JLD2, DataFrames, CSV, Printf
using Statistics, LinearAlgebra, SparseArrays
using SignalAnalysis, Optim, AdaptiveEstimators
using ..LDPCJDPMemoized
using ..SignalUtils

export run_dfec_experiment, DataCache

# ============================================================================
# Data Caching Structure
# ============================================================================
"""
    DataCache

Holds preprocessed data to avoid recomputation.
"""
mutable struct DataCache
    y_train_matrix::Matrix  # Can be ComplexF64 or Float64
    packet_matrix::Matrix   # Can be ComplexF64 or Float64
    H_est_omp::Matrix{ComplexF64}
    H_est_mmse::Matrix{ComplexF64}
    x_train::Vector{ComplexF64}
    x_datas::Matrix         # Can be ComplexF64 or Float64
    h_full::Vector{ComplexF64}
    h_pos::Vector{Int}
    h_init::Vector{ComplexF64}
    num_frames::Int
    y_train_frames::Vector{Int}
    packet_frames::Vector{Int}
    packet_blocks::Vector{Int}
    packet_to_train_idx::Vector{Int}
    packet_to_xdata_idx::Vector{Int}
end

const METADATA_COLS = (:frame, :block)

# ============================================================================
# Initialization Functions
# ============================================================================
"""
    load_data(data_dir::String, data_file::String, signal_file::String)

Load experimental data from files.
"""
function load_data(data_dir::String, data_file::String, signal_file::String)
    # Convert to absolute path (relative to project root)
    if !isabspath(data_dir)
        # @__DIR__ is src/, go up to project root, then add data_dir
        project_root = dirname(@__DIR__)
        data_dir = joinpath(project_root, data_dir)
    end

    jld2_path = joinpath(data_dir, data_file)
    signal_path = joinpath(data_dir, signal_file)

    @load jld2_path all_ytrain_df all_packets_df
    # Note: signal_path exists but is not used in core pipeline
    # SS = SignalAnalysis.read(signal_path)  # Can be loaded if needed

    return (all_ytrain_df, all_packets_df, nothing)
end

"""
    prepare_training_signals(num_train::Int)

Generate training sequence.
"""
function prepare_training_signals(num_train::Int)
    x_train = convert.(ComplexF64, repeat(mseq(8), num_train))
    return x_train
end

"""
    prepare_data_signals(d_nodes::Int, t_nodes::Int, npc::Int, num_train::Int, num_data::Int,
                         gap::Int, num_frame_repeats::Int, data_dir::String)

Generate data packet structure.
"""
function prepare_data_signals(d_nodes::Int, t_nodes::Int, npc::Int, num_train::Int, num_data::Int,
                              gap::Int, num_frame_repeats::Int, data_dir::String)
    # Change to data directory to load LDPC files, then change back
    old_dir = pwd()
    try
        cd(data_dir)
        code, _, _, _ = LDPCJDPMemoized.initcode(d_nodes, t_nodes, npc)
        _, x_datas, _ = LDPCJDPMemoized.makepacket(code, num_train, num_data, gap)
        x_datas = repeat(x_datas, num_frame_repeats, 1)
        return x_datas
    finally
        cd(old_dir)
    end
end

metadata_columns(df) = [col for col in METADATA_COLS if col in propertynames(df)]

function extract_signal_matrix(df)
    return Matrix(select(df, Not(metadata_columns(df))))
end

function extract_row_metadata(df)
    frames = :frame in propertynames(df) ? Int.(df.frame) : collect(1:nrow(df))
    blocks = :block in propertynames(df) ? Int.(df.block) : zeros(Int, nrow(df))
    return frames, blocks
end

function build_packet_mappings(all_ytrain_df, all_packets_df, x_datas)
    y_train_frames, _ = extract_row_metadata(all_ytrain_df)
    packet_frames, packet_blocks = extract_row_metadata(all_packets_df)

    frame_to_train_idx = Dict(frame => idx for (idx, frame) in enumerate(y_train_frames))
    packet_to_train_idx = Vector{Int}(undef, length(packet_frames))
    for i in eachindex(packet_frames)
        packet_to_train_idx[i] = get(frame_to_train_idx, packet_frames[i], 0)
        packet_to_train_idx[i] == 0 && error("No training row found for packet frame $(packet_frames[i])")
    end

    blocks_per_frame, remainder = divrem(size(x_datas, 1), length(y_train_frames))
    remainder == 0 || error("x_datas row count $(size(x_datas, 1)) is not divisible by training rows $(length(y_train_frames))")

    expected_blocks = collect(1:blocks_per_frame)
    observed_blocks = sort(unique(packet_blocks))
    observed_blocks == expected_blocks || error("Expected packet block IDs $(expected_blocks), found $(observed_blocks)")

    packet_to_xdata_idx = Vector{Int}(undef, length(packet_frames))
    for i in eachindex(packet_frames)
        packet_to_xdata_idx[i] = (packet_to_train_idx[i] - 1) * blocks_per_frame + packet_blocks[i]
    end

    return y_train_frames, packet_frames, packet_blocks, packet_to_train_idx, packet_to_xdata_idx
end

"""
    estimate_channels(y_train_matrix::Matrix, x_train::Vector, h_len::Int, k_sparse::Int, σ²::Float64)

Estimate channels using OMP and MMSE methods.
"""
function estimate_channels(y_train_matrix::Matrix, x_train::Vector, h_len::Int, k_sparse::Int, σ²::Float64)
    num_frames = size(y_train_matrix, 1)
    H_est_omp = zeros(ComplexF64, num_frames, h_len)
    H_est_mmse = zeros(ComplexF64, num_frames, h_len)

    for i in 1:num_frames
        y_i = convert.(ComplexF64, y_train_matrix[i, 1:length(x_train)])
        H_est_omp[i, :] .= SignalUtils.estimate_omp_channel(y_i, x_train, h_len, k_sparse)
        H_est_mmse[i, :] .= SignalUtils.estimate_mmse_channel(y_i, x_train, h_len; σ²=σ²)
    end

    return (H_est_omp, H_est_mmse)
end

"""
    prepare_cache(all_ytrain_df, all_packets_df, x_train, x_datas, H_est_omp, H_est_mmse)

Prepare data cache for fast access during experiment.
"""
function prepare_cache(all_ytrain_df, all_packets_df, x_train, x_datas, H_est_omp, H_est_mmse)
    y_train_matrix = extract_signal_matrix(all_ytrain_df)
    packet_matrix = extract_signal_matrix(all_packets_df)
    y_train_frames, packet_frames, packet_blocks, packet_to_train_idx, packet_to_xdata_idx =
        build_packet_mappings(all_ytrain_df, all_packets_df, x_datas)

    h_full = H_est_omp[1, :]
    h_pos = findall(!iszero, h_full)
    h_init = h_full[h_pos]

    num_frames = size(y_train_matrix, 1)
    size(x_datas, 1) == size(packet_matrix, 1) ||
        error("x_datas rows ($(size(x_datas, 1))) do not match packet rows ($(size(packet_matrix, 1)))")

    return DataCache(
        y_train_matrix, packet_matrix, H_est_omp, H_est_mmse,
        x_train, x_datas, h_full, h_pos, h_init, num_frames,
        y_train_frames, packet_frames, packet_blocks, packet_to_train_idx, packet_to_xdata_idx
    )
end

function dfe_equalize_packet(cache::DataCache, train_idx::Int, xdata_idx::Int, y_data,
                             tr_len::Int, h_len::Int, Q::Vector, pilot_len::Int)
    n_code = length(y_data)
    y_train = cache.y_train_matrix[train_idx, 1:length(cache.x_train)]

    r = fit!(DFE(h_len), RLS(),
             [y_train[1:tr_len]; y_data],
             [cache.x_train[1:tr_len]; cache.x_datas[xdata_idx, 1:pilot_len]],
             tr_len + n_code, nearest(Q))

    x_hat = r.y[tr_len+1:end]
    x_ref = cache.x_datas[xdata_idx, 1:n_code]
    (_, ph) = SignalUtils.argminphase(x_hat, x_ref)
    return x_hat .* exp(-im * deg2rad(-ph))
end

function warm_start_logits(x_hat, pilot::Vector{Int}, x_pilot; clip::Float64=0.98, pilot_boost::Float64=4.0)
    z0 = atanh.(clamp.(real.(x_hat), -clip, clip))
    if !isempty(pilot)
        pilot_score = sum(sign.(real.(x_hat[pilot])) .* sign.(real.(x_pilot)))
        if pilot_score < 0
            z0 .*= -1
        end
        z0[pilot] .= pilot_boost .* sign.(real.(x_pilot))
    end
    return z0
end

# ============================================================================
# Main Processing Loop
# ============================================================================
"""
    process_pilot_fraction(cache::DataCache, pilot_frac::Float64, frame_nums::Vector{Int},
                           d_nodes::Int, t_nodes::Int, npc::Int, n::Int, tr_len::Int,
                           h_len::Int, λ::Float64, γ::Float64, η::Float64, Q::Vector, data_dir::String)

Process a single pilot fraction across multiple frames.
"""
function process_pilot_fraction(cache::DataCache, pilot_frac::Float64, frame_nums::Vector{Int},
                                d_nodes::Int, t_nodes::Int, npc::Int, n::Int, tr_len::Int,
                                h_len::Int, λ::Float64, γ::Float64, η::Float64, Q::Vector, data_dir::String)

    results = DataFrame(
        frame=Int[], block=Int[], ber_grad=Float64[], ber_dfe=Float64[],
        ber_spa=Float64[], ber_min=Float64[], pilot_frac=Float64[]
    )

    println("\n🌐 Testing pilot_row_fraction = $(round(pilot_frac, digits=2))")

    # Load code with proper directory context
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

    for packet_idx in frame_nums
        1 <= packet_idx <= size(cache.packet_matrix, 1) ||
            error("Packet index $(packet_idx) is out of bounds for $(size(cache.packet_matrix, 1)) packets")

        train_idx = cache.packet_to_train_idx[packet_idx]
        xdata_idx = cache.packet_to_xdata_idx[packet_idx]
        frame_id = cache.packet_frames[packet_idx]
        block_id = cache.packet_blocks[packet_idx]

        @printf "🧾 Frame %d Block %d (row %d)\n" frame_id block_id packet_idx

        y_data = cache.packet_matrix[packet_idx, 1:n_code]

        # === Step 1: DFE Equalization ===
        x̂ = dfe_equalize_packet(cache, train_idx, xdata_idx, y_data, tr_len, h_len, Q, p_len)
        x = cache.x_datas[xdata_idx, 1:end]
        dfe_ber = sum(abs, sign.(real.(x̂[p_len:end])) .!= sign.(real.(x[p_len:end]))) / n_code

        # === Step 2: SPA Decoding ===
        L_ch = 2 .* real.(x̂)
        H_sparse = LDPCJDPMemoized.get_H_sparse(code)
        x_spa, spa_iters = LDPCJDPMemoized.sum_product_decode(H_sparse, real.(x̂), 1.0, idrows, cols)
        x_spa = x_spa[p_len+1:end]
        x = x[p_len+1:end]
        d = LDPCJDPMemoized.demodulate.(x)
        spa_ber = sum(x_spa .== d) / length(x_spa)

        # === Step 3: Joint Channel and Symbol Decoding ===
        h_full = cache.H_est_omp[train_idx, :]
        h_pos = findall(!iszero, h_full)
        h_init = h_full[h_pos]
        x_pilot = cache.x_datas[xdata_idx, pilot]
        pilot_data = LDPCJDPMemoized.demodulate.(x_pilot)
        pilot_bpsk = LDPCJDPMemoized.modulate.(pilot_data)
        z_init = warm_start_logits(x̂, pilot, x_pilot)

        x̂_bits, h_est, result = LDPCJDPMemoized.decode_sparse_joint(
            y_data, code, idrows, pilot, pilot_bpsk, h_pos;
            h_init=h_init, z_init=z_init, λ=λ, γ=γ, η=η, verbose=false
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

        min_ber = min(spa_ber, dfe_ber)

        push!(results, (
            frame=frame_id, block=block_id,
            ber_grad=ber, ber_dfe=dfe_ber, ber_spa=spa_ber, ber_min=min_ber,
            pilot_frac=pilot_frac
        ))

        @printf "Frame %d Block %d | DFEC BER = %.4f | DFE+FEC BER %.4f\n" frame_id block_id ber min_ber
    end

    return results
end

"""
    run_dfec_experiment(config_file::String)

Main entry point: run full DFEC experiment.
"""
function run_dfec_experiment(config_file::String)
    include(config_file)

    # Resolve data directory path
    data_dir = DATA_DIR
    if !isabspath(data_dir)
        project_root = dirname(@__DIR__)
        data_dir = joinpath(project_root, data_dir)
    end

    println("\n" * "="^70)
    println("╔═══════════════════════════════════════════════════════════════╗")
    println("║         DFEC Experiment Configuration                          ║")
    println("╚═══════════════════════════════════════════════════════════════╝")
    println("🔹 Frames to process: $(NUM_FRAMES_TO_PROCESS)")
    println("🔹 Pilot fractions tested: $(PILOT_FRACTIONS)")
    println("="^70 * "\n")

    # === Load Data ===
    println("📂 Loading data...")
    all_ytrain_df, all_packets_df, _ = load_data(data_dir, DATA_FILE, SIGNAL_FILE)

    num_frame_repeats = nrow(all_ytrain_df)
    x_train = prepare_training_signals(NUM_TRAIN)
    x_datas = prepare_data_signals(D_NODES, T_NODES, NPC, NUM_TRAIN, NUM_DATA, GAP, num_frame_repeats, data_dir)

    tr_len = length(x_train)
    n = CODE_N

    # === Estimate Channels ===
    println("🔍 Estimating channels...")
    H_est_omp, H_est_mmse = estimate_channels(
        extract_signal_matrix(all_ytrain_df),
        x_train, H_LEN, K_SPARSE, NOISE_VARIANCE
    )

    # === Prepare Cache ===
    println("💾 Preparing data cache...")
    cache = prepare_cache(all_ytrain_df, all_packets_df, x_train, x_datas, H_est_omp, H_est_mmse)

    # === Run Experiment ===
    println("⚙️  Starting experiment...\n")

    all_results = DataFrame(
        frame=Int[], block=Int[], ber_grad=Float64[], ber_dfe=Float64[],
        ber_spa=Float64[], ber_min=Float64[], pilot_frac=Float64[]
    )

    packet_rows_to_process = collect(1:min(NUM_FRAMES_TO_PROCESS, size(cache.packet_matrix, 1)))

    for pilot_frac in PILOT_FRACTIONS
        results = process_pilot_fraction(
            cache, pilot_frac, packet_rows_to_process,
            D_NODES, T_NODES, NPC, n, tr_len, H_LEN, LAMBDA, GAMMA, ETA, BPSK_CONSTELLATION, data_dir
        )
        append!(all_results, results)
    end

    # === Save Results ===
    results_dir = RESULTS_DIR
    if !isabspath(results_dir)
        project_root = dirname(@__DIR__)
        results_dir = joinpath(project_root, results_dir)
    end
    mkpath(results_dir)  # Create if doesn't exist
    results_path = joinpath(results_dir, RESULTS_FILE)
    CSV.write(results_path, all_results)
    println("\n✅ Results saved to: $(results_path)")
    println("   Rows: $(nrow(all_results))")
    println("\n" * "="^70)

    return all_results
end

end # module
