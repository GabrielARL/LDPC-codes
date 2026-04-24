"""
    sync_alignment_check.jl

Reference-based alignment check for packets 19 and 20.

This script does not compare adjacent packets directly, because each packet
contains different data. Instead it compares the two valid 128-sample windows
inside each stored 129-sample packet (`s1:s128` and `s2:s129`) and measures
which window decodes better under the current receiver model.
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

const PACKET_ROWS = (19, 20)

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

function analyze_packet(cache::ExperimentCore.DataCache, packet_idx::Int, data_dir::String)
    rows = NamedTuple[]
    for pilot_frac in Float64.(collect(PILOT_FRACTIONS))
        p_len = load_pilot_length(data_dir, pilot_frac)
        ber_1 = dfe_window_ber(cache, packet_idx, 1, p_len)
        ber_2 = dfe_window_ber(cache, packet_idx, 2, p_len)
        best_window = ber_1 <= ber_2 ? 1 : 2
        push!(rows, (
            pilot_frac = pilot_frac,
            ber_window_1 = ber_1,
            ber_window_2 = ber_2,
            best_window = best_window,
            improvement = ber_1 - ber_2
        ))
    end
    return rows
end

function summarize_packet(rows)
    best_windows = [row.best_window for row in rows]
    return (
        window_1_wins = count(==(1), best_windows),
        window_2_wins = count(==(2), best_windows),
        mean_improvement = mean(row.improvement for row in rows)
    )
end

data_dir = resolve_data_dir()
cache = build_cache(data_dir)

println("="^78)
println("REFERENCE-BASED ALIGNMENT CHECK: PACKETS 19 AND 20")
println("="^78 * "\n")
println("Method:")
println("  Compare `s1:s128` versus `s2:s129` for each packet row.")
println("  Lower DFE BER means the corresponding 128-sample window is better aligned.\n")

packet_results = Dict{Int, Any}()
for packet_idx in PACKET_ROWS
    frame_id = cache.packet_frames[packet_idx]
    block_id = cache.packet_blocks[packet_idx]
    train_idx = cache.packet_to_train_idx[packet_idx]

    println("Packet row $(packet_idx): frame=$(frame_id), block=$(block_id), train_row=$(train_idx)")
    println("┌────────────┬──────────────┬──────────────┬──────────────┬────────────┐")
    println("│ Pilot frac │ BER s1:s128  │ BER s2:s129  │ Improvement  │ Best win   │")
    println("├────────────┼──────────────┼──────────────┼──────────────┼────────────┤")

    rows = analyze_packet(cache, packet_idx, data_dir)
    packet_results[packet_idx] = rows

    for row in rows
        @printf(
            "│ %10.2f │ %12.4f │ %12.4f │ %12.4f │ %10d │\n",
            row.pilot_frac, row.ber_window_1, row.ber_window_2, row.improvement, row.best_window
        )
    end
    println("└────────────┴──────────────┴──────────────┴──────────────┴────────────┘")

    summary = summarize_packet(rows)
    println(
        "Summary: window1 wins=$(summary.window_1_wins), " *
        "window2 wins=$(summary.window_2_wins), " *
        "mean(ber1-ber2)=$(round(summary.mean_improvement, digits=4))\n"
    )
end

summary_19 = summarize_packet(packet_results[19])
summary_20 = summarize_packet(packet_results[20])

println("="^78)
println("DIAGNOSIS")
println("="^78 * "\n")

if summary_19.window_2_wins == 0 && summary_20.window_2_wins == 0
    println("Neither packet prefers the shifted window.")
    println("Conclusion: no evidence of a 19→20 alignment issue.")
elseif summary_19.window_2_wins > 0 && summary_20.window_2_wins > 0
    println("Both packets sometimes prefer the shifted window.")
    println("Conclusion: this looks like a local/shared window preference, not a special 19→20 boundary slip.")
elseif summary_19.window_2_wins > 0
    println("Packet 19 prefers the shifted window more often than packet 20.")
    println("Conclusion: any one-sample preference is not unique to the packet-20 boundary.")
else
    println("Packet 20 prefers the shifted window more often than packet 19.")
    println("Conclusion: investigate packet 20 further, but this is still a one-sample window test, not a 28-29 sample slip.")
end

println("\nNote:")
println("  The earlier cross-packet boundary detector compared unrelated payloads.")
println("  This script is a direct alignment check against the expected transmitted block.")
