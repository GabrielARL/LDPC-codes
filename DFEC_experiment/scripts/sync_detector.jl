"""
    sync_detector.jl

Reference-based packet window detector.

For each packet row, this script compares the two valid 128-sample windows inside
the stored 129-sample packet (`s1:s128` and `s2:s129`) and records which window
decodes better under the current receiver model.

This replaces the old boundary cross-correlation approach, which was comparing
adjacent packets with different payloads and could produce false "misalignment"
alarms.
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

using AdaptiveEstimators, JLD2, DataFrames, Printf, Statistics

const ANALYSIS_PILOT_FRACTION = 0.41
const NUM_PACKETS_TO_ANALYZE = 100

function resolve_data_dir()
    data_dir = DATA_DIR
    if !isabspath(data_dir)
        project_root = dirname(@__DIR__)
        data_dir = joinpath(project_root, data_dir)
    end
    return data_dir
end

function build_cache(data_dir::String)
    jld2_path = joinpath(data_dir, DATA_FILE)
    @load jld2_path all_ytrain_df all_packets_df

    x_train = ExperimentCore.prepare_training_signals(NUM_TRAIN)
    x_datas = ExperimentCore.prepare_data_signals(
        D_NODES, T_NODES, NPC, NUM_TRAIN, NUM_DATA, GAP, nrow(all_ytrain_df), data_dir
    )
    H_est_omp, H_est_mmse = ExperimentCore.estimate_channels(
        ExperimentCore.extract_signal_matrix(all_ytrain_df),
        x_train, H_LEN, K_SPARSE, NOISE_VARIANCE
    )

    cache = ExperimentCore.prepare_cache(
        all_ytrain_df, all_packets_df, x_train, x_datas, H_est_omp, H_est_mmse
    )
    return cache
end

function load_pilot_length(data_dir::String, pilot_frac::Float64)
    old_dir = pwd()
    try
        cd(data_dir)
        _, _, _, pilot = LDPCJDPMemoized.initcode(
            D_NODES, T_NODES, NPC; pilot_row_fraction=pilot_frac
        )
        return length(sort(pilot))
    finally
        cd(old_dir)
    end
end

function dfe_window_ber(cache::ExperimentCore.DataCache, packet_idx::Int, start_idx::Int, p_len::Int)
    stop_idx = start_idx + CODE_N - 1
    stop_idx <= size(cache.packet_matrix, 2) ||
        error("Requested window $(start_idx):$(stop_idx) exceeds packet width $(size(cache.packet_matrix, 2))")

    train_idx = cache.packet_to_train_idx[packet_idx]
    xdata_idx = cache.packet_to_xdata_idx[packet_idx]
    tr_len = length(cache.x_train)

    y_train = cache.y_train_matrix[train_idx, 1:tr_len]
    y_data = cache.packet_matrix[packet_idx, start_idx:stop_idx]

    r = fit!(
        DFE(H_LEN), RLS(),
        [y_train; y_data],
        [cache.x_train; cache.x_datas[xdata_idx, 1:p_len]],
        tr_len + CODE_N,
        nearest(BPSK_CONSTELLATION)
    )

    x_hat = r.y[tr_len+1:end]
    x_ref = cache.x_datas[xdata_idx, :]
    (_, ph) = SignalUtils.argminphase(x_hat, x_ref)
    x_hat = x_hat .* exp(-im * deg2rad(-ph))

    return sum(abs, sign.(real.(x_hat[p_len:end])) .!= sign.(real.(x_ref[p_len:end]))) / CODE_N
end

function analyze_packets(cache::ExperimentCore.DataCache, data_dir::String)
    p_len = load_pilot_length(data_dir, ANALYSIS_PILOT_FRACTION)
    n_packets = min(NUM_PACKETS_TO_ANALYZE, size(cache.packet_matrix, 1))

    results = DataFrame(
        packet_row = Int[],
        frame = Int[],
        block = Int[],
        ber_window_1 = Float64[],
        ber_window_2 = Float64[],
        delta = Float64[],
        best_window = Int[]
    )

    for packet_idx in 1:n_packets
        ber_1 = dfe_window_ber(cache, packet_idx, 1, p_len)
        ber_2 = dfe_window_ber(cache, packet_idx, 2, p_len)
        best_window = ber_1 <= ber_2 ? 1 : 2

        push!(results, (
            packet_row = packet_idx,
            frame = cache.packet_frames[packet_idx],
            block = cache.packet_blocks[packet_idx],
            ber_window_1 = ber_1,
            ber_window_2 = ber_2,
            delta = ber_2 - ber_1,
            best_window = best_window
        ))
    end

    return results
end

data_dir = resolve_data_dir()
cache = build_cache(data_dir)
results = analyze_packets(cache, data_dir)

println("="^78)
println("REFERENCE-BASED WINDOW DETECTOR")
println("="^78 * "\n")
println("Analysis pilot fraction: $(ANALYSIS_PILOT_FRACTION)")
println("Packets analyzed: $(nrow(results))\n")

window_2_rows = filter(:best_window => ==(2), results)
println("Overall summary:")
println("├─ Window 1 preferred: $(count(==(1), results.best_window))")
println("├─ Window 2 preferred: $(count(==(2), results.best_window))")
println("└─ Mean delta (ber2 - ber1): $(round(mean(results.delta), digits=4))\n")

println("Top packet rows favoring window 2:")
println("┌──────────┬────────┬───────┬──────────────┬──────────────┬────────────┐")
println("│ Row      │ Frame  │ Block │ BER s1:s128  │ BER s2:s129  │ Delta      │")
println("├──────────┼────────┼───────┼──────────────┼──────────────┼────────────┤")
for row in eachrow(first(sort(window_2_rows, :delta), min(15, nrow(window_2_rows))))
    @printf(
        "│ %8d │ %6d │ %5d │ %12.4f │ %12.4f │ %10.4f │\n",
        row.packet_row, row.frame, row.block, row.ber_window_1, row.ber_window_2, row.delta
    )
end
println("└──────────┴────────┴───────┴──────────────┴──────────────┴────────────┘\n")

block_summary = combine(
    groupby(results, :block),
    nrow => :count,
    :best_window => (x -> count(==(2), x)) => :window_2_count,
    :delta => mean => :mean_delta
)

println("Block summary:")
println("┌───────┬───────┬────────────────┬────────────┐")
println("│ Block │ Count │ Window2 better │ Mean delta │")
println("├───────┼───────┼────────────────┼────────────┤")
for row in eachrow(block_summary)
    @printf(
        "│ %5d │ %5d │ %14d │ %10.4f │\n",
        row.block, row.count, row.window_2_count, row.mean_delta
    )
end
println("└───────┴───────┴────────────────┴────────────┘\n")

println("Focus blocks 18-20:")
for row in eachrow(filter(:block => in(18:20), block_summary))
    println(
        "  block $(row.block): window2 better in $(row.window_2_count)/$(row.count) packets, " *
        "mean delta=$(round(row.mean_delta, digits=4))"
    )
end

println("\nInterpretation:")
println("  Negative delta means the shifted window (`s2:s129`) decodes better.")
println("  Large or systematic negative deltas are the meaningful evidence to watch.")
