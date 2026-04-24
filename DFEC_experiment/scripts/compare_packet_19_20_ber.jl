"""
    compare_packet_19_20_ber.jl

Run packets 19 and 20 through the current DFE path and compare BERs.

For each configured pilot fraction, this script evaluates both valid 128-sample
windows inside the stored 129-sample packet:

  - `s1:s128`
  - `s2:s129`

and prints the BERs for packet rows 19 and 20 side-by-side.

Run with:

    julia --project=. scripts/compare_packet_19_20_ber.jl
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

using AdaptiveEstimators, JLD2, DataFrames, Printf

const PACKET_A = 19
const PACKET_B = 20

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

    return ExperimentCore.prepare_cache(
        all_ytrain_df, all_packets_df, x_train, x_datas, H_est_omp, H_est_mmse
    )
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

function packet_label(cache::ExperimentCore.DataCache, packet_idx::Int)
    return "row=$(packet_idx) frame=$(cache.packet_frames[packet_idx]) block=$(cache.packet_blocks[packet_idx])"
end

data_dir = resolve_data_dir()
cache = build_cache(data_dir)

println("="^88)
println("PACKET 19 VS 20 BER COMPARISON")
println("="^88 * "\n")
println("Packet 19: $(packet_label(cache, PACKET_A))")
println("Packet 20: $(packet_label(cache, PACKET_B))\n")

println("┌────────────┬──────────────┬──────────────┬──────────────┬──────────────┬────────────┬────────────┐")
println("│ Pilot frac │ P19 s1:s128  │ P19 s2:s129  │ P20 s1:s128  │ P20 s2:s129  │ P20-P19 s1 │ P20-P19 s2 │")
println("├────────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────┼────────────┤")

for pilot_frac in Float64.(collect(PILOT_FRACTIONS))
    p_len = load_pilot_length(data_dir, pilot_frac)

    p19_w1 = dfe_window_ber(cache, PACKET_A, 1, p_len)
    p19_w2 = dfe_window_ber(cache, PACKET_A, 2, p_len)
    p20_w1 = dfe_window_ber(cache, PACKET_B, 1, p_len)
    p20_w2 = dfe_window_ber(cache, PACKET_B, 2, p_len)

    @printf(
        "│ %10.2f │ %12.4f │ %12.4f │ %12.4f │ %12.4f │ %10.4f │ %10.4f │\n",
        pilot_frac, p19_w1, p19_w2, p20_w1, p20_w2, p20_w1 - p19_w1, p20_w2 - p19_w2
    )
end

println("└────────────┴──────────────┴──────────────┴──────────────┴──────────────┴────────────┴────────────┘\n")

println("Interpretation:")
println("  Lower BER is better.")
println("  `P20-P19` > 0 means packet 20 is worse than packet 19 for that window.")
println("  If both packets prefer `s2:s129`, that suggests a shared one-sample preference,")
println("  not a special packet-20-only boundary slip.")
