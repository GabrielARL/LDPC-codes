"""
    run_phase_corrected_900.jl

Run the DFEC decoding pipeline on all 900 packet rows using the phase-corrected
JLD2 dataset and save the results to a separate CSV.

Defaults:
  - Input JLD2:  data/logged_packets_and_ytrain_phase_corrected.jld2
  - Output CSV:  results/ldpc_ber_phase_corrected_900.csv
  - Packet rows: 1:900
  - Pilot fracs: config default
  - Chunk size:  25 packet rows per checkpoint

Usage:

    julia --project=. scripts/run_phase_corrected_900.jl

Optional custom output CSV path:

    julia --project=. scripts/run_phase_corrected_900.jl /tmp/phase_corrected_900.csv

You can also run a subset or resume an interrupted run:

    julia --project=. scripts/run_phase_corrected_900.jl --pilot 0.41 --start 1 --end 900 --chunk-size 25 --resume
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

using JLD2, CSV, DataFrames, Statistics

const PHASE_CORRECTED_JLD2 = "logged_packets_and_ytrain_phase_corrected.jld2"
const DEFAULT_OUTPUT_CSV = "ldpc_ber_phase_corrected_900.csv"
const DEFAULT_CHUNK_SIZE = 25

Base.@kwdef mutable struct RunOptions
    output_csv::String
    packet_start::Int = 1
    packet_end::Int = 0
    pilot_fracs::Vector{Float64} = collect(PILOT_FRACTIONS)
    chunk_size::Int = DEFAULT_CHUNK_SIZE
    resume::Bool = false
end

function resolve_project_paths()
    project_root = dirname(@__DIR__)
    data_dir = isabspath(DATA_DIR) ? DATA_DIR : joinpath(project_root, DATA_DIR)
    results_dir = isabspath(RESULTS_DIR) ? RESULTS_DIR : joinpath(project_root, RESULTS_DIR)
    return project_root, data_dir, results_dir
end

function resolve_path_arg(path::String, default_root::String)
    return isabspath(path) ? path : joinpath(default_root, path)
end

function parse_args(results_dir::String, num_packet_rows::Int)
    opts = RunOptions(output_csv=joinpath(results_dir, DEFAULT_OUTPUT_CSV), packet_end=num_packet_rows)

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--output"
            i += 1
            i <= length(ARGS) || error("Missing value after --output")
            opts.output_csv = resolve_path_arg(ARGS[i], pwd())
        elseif arg == "--start"
            i += 1
            i <= length(ARGS) || error("Missing value after --start")
            opts.packet_start = parse(Int, ARGS[i])
        elseif arg == "--end"
            i += 1
            i <= length(ARGS) || error("Missing value after --end")
            opts.packet_end = parse(Int, ARGS[i])
        elseif arg == "--pilot"
            i += 1
            i <= length(ARGS) || error("Missing value after --pilot")
            opts.pilot_fracs = [parse(Float64, ARGS[i])]
        elseif arg == "--chunk-size"
            i += 1
            i <= length(ARGS) || error("Missing value after --chunk-size")
            opts.chunk_size = parse(Int, ARGS[i])
        elseif arg == "--resume"
            opts.resume = true
        elseif startswith(arg, "--")
            error("Unknown option: $(arg)")
        elseif i == 1
            opts.output_csv = resolve_path_arg(arg, pwd())
        else
            error("Unexpected positional argument: $(arg)")
        end
        i += 1
    end

    1 <= opts.packet_start <= num_packet_rows || error("--start must be in 1:$(num_packet_rows)")
    1 <= opts.packet_end <= num_packet_rows || error("--end must be in 1:$(num_packet_rows)")
    opts.packet_start <= opts.packet_end || error("--start must be <= --end")
    opts.chunk_size > 0 || error("--chunk-size must be positive")

    return opts
end

function empty_results_df()
    return DataFrame(
        frame=Int[], block=Int[], ber_grad=Float64[], ber_dfe=Float64[],
        ber_spa=Float64[], ber_min=Float64[], pilot_frac=Float64[]
    )
end

function load_existing_results(output_csv::String, resume::Bool)
    if resume && isfile(output_csv)
        return CSV.read(output_csv, DataFrame)
    end
    return empty_results_df()
end

pilot_key(pilot_frac::Real) = round(Float64(pilot_frac), digits=4)

function completed_result_keys(results::DataFrame)
    return Set{Tuple{Int, Int, Float64}}(
        (Int(row.frame), Int(row.block), pilot_key(row.pilot_frac))
        for row in eachrow(results)
    )
end

function pending_packet_rows(cache::ExperimentCore.DataCache, packet_rows::Vector{Int},
                             completed_keys::Set{Tuple{Int, Int, Float64}}, pilot_frac::Float64)
    pkey = pilot_key(pilot_frac)
    return [
        packet_idx for packet_idx in packet_rows
        if (cache.packet_frames[packet_idx], cache.packet_blocks[packet_idx], pkey) ∉ completed_keys
    ]
end

function sort_results!(results::DataFrame)
    sort!(results, [:pilot_frac, :frame, :block])
    return results
end

function main()
    project_root, data_dir, results_dir = resolve_project_paths()
    input_jld2 = joinpath(data_dir, PHASE_CORRECTED_JLD2)

    isfile(input_jld2) || error("Input JLD2 not found: $(input_jld2)")

    println("📂 Loading corrected dataset...")
    @load input_jld2 all_ytrain_df all_packets_df

    num_packet_rows = nrow(all_packets_df)
    opts = parse_args(results_dir, num_packet_rows)
    packet_rows = collect(opts.packet_start:opts.packet_end)
    existing_results = load_existing_results(opts.output_csv, opts.resume)

    println("\n" * "="^88)
    println("DFEC RUN ON PHASE-CORRECTED DATA")
    println("="^88)
    println("Input JLD2:   $(input_jld2)")
    println("Output CSV:   $(opts.output_csv)")
    println("Packet rows:  $(first(packet_rows)):$(last(packet_rows)) ($(length(packet_rows)) rows)")
    println("Pilot fracs:  $(opts.pilot_fracs)")
    println("Chunk size:   $(opts.chunk_size)")
    println("Resume mode:  $(opts.resume)")
    println("="^88 * "\n")

    println("   Training rows: $(nrow(all_ytrain_df))")
    println("   Packet rows:   $(num_packet_rows)")

    x_train = ExperimentCore.prepare_training_signals(NUM_TRAIN)
    x_datas = ExperimentCore.prepare_data_signals(
        D_NODES, T_NODES, NPC, NUM_TRAIN, NUM_DATA, GAP, nrow(all_ytrain_df), data_dir
    )
    tr_len = length(x_train)

    println("🔍 Estimating channels...")
    y_train_matrix = ExperimentCore.extract_signal_matrix(all_ytrain_df)
    H_est_omp, H_est_mmse = ExperimentCore.estimate_channels(
        y_train_matrix, x_train, H_LEN, K_SPARSE, NOISE_VARIANCE
    )

    println("💾 Preparing cache...")
    cache = ExperimentCore.prepare_cache(
        all_ytrain_df, all_packets_df, x_train, x_datas, H_est_omp, H_est_mmse
    )

    all_results = existing_results
    completed_keys = completed_result_keys(all_results)

    if !isempty(all_results)
        println("📄 Existing results loaded: $(nrow(all_results)) rows")
    end

    for pilot_frac in opts.pilot_fracs
        pending_rows = pending_packet_rows(cache, packet_rows, completed_keys, pilot_frac)
        if isempty(pending_rows)
            println("⏭️  Pilot fraction $(round(pilot_frac, digits=2)) already complete for requested packet rows.")
            continue
        end

        println("⚙️  Running pilot fraction $(round(pilot_frac, digits=2)) over $(length(pending_rows)) pending packets...")

        chunk_counter = 0
        for chunk in Iterators.partition(pending_rows, opts.chunk_size)
            chunk_counter += 1
            chunk_rows = collect(chunk)
            println(
                "   • Chunk $(chunk_counter): packet rows $(first(chunk_rows))-$(last(chunk_rows)) " *
                "($(length(chunk_rows)) rows)"
            )

            results = redirect_stdout(devnull) do
                ExperimentCore.process_pilot_fraction(
                    cache, pilot_frac, chunk_rows,
                    D_NODES, T_NODES, NPC, CODE_N, tr_len, H_LEN, LAMBDA, GAMMA, ETA,
                    BPSK_CONSTELLATION, data_dir
                )
            end

            append!(all_results, results)
            sort_results!(all_results)
            completed_keys = completed_result_keys(all_results)

            mkpath(dirname(opts.output_csv))
            CSV.write(opts.output_csv, all_results)

            println(
                "     saved $(nrow(all_results)) total rows | chunk mean DFEC BER=$(round(mean(results.ber_grad), digits=4)) | " *
                "chunk mean DFE+FEC BER=$(round(mean(results.ber_min), digits=4))"
            )
        end
    end

    println("\n✅ Finished")
    println("   Rows written: $(nrow(all_results))")
    println("   Unique frames: $(length(unique(all_results.frame)))")
    println("   Mean DFEC BER: $(round(mean(all_results.ber_grad), digits=4))")
    println("   Mean DFE+FEC BER: $(round(mean(all_results.ber_min), digits=4))")
    println("   CSV: $(opts.output_csv)")
end

main()
