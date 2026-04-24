#!/usr/bin/env julia

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "config"))

include("../config/experiment_config.jl")
include("../src/SignalUtils.jl")
include("../src/LDPCJDPMemoized.jl")
include("../src/ExperimentCore.jl")

using .SignalUtils
using .LDPCJDPMemoized
using .ExperimentCore
using CSV
using DataFrames
using Printf
using Statistics

Base.@kwdef struct Options
    pilot_fracs::Vector{Float64} = Float64.(collect(PILOT_FRACTIONS))
    bundle_dir::String = ""
    julia_csv::String = ""
    cpp_csv::String = ""
    merged_csv::String = ""
    summary_csv::String = ""
    skip_export::Bool = false
    skip_build_cpp::Bool = false
end

function replace_opts(opts::Options; pilot_fracs::Vector{Float64}=opts.pilot_fracs,
                      bundle_dir::String=opts.bundle_dir,
                      julia_csv::String=opts.julia_csv,
                      cpp_csv::String=opts.cpp_csv,
                      merged_csv::String=opts.merged_csv,
                      summary_csv::String=opts.summary_csv,
                      skip_export::Bool=opts.skip_export,
                      skip_build_cpp::Bool=opts.skip_build_cpp)
    return Options(
        pilot_fracs=pilot_fracs,
        bundle_dir=bundle_dir,
        julia_csv=julia_csv,
        cpp_csv=cpp_csv,
        merged_csv=merged_csv,
        summary_csv=summary_csv,
        skip_export=skip_export,
        skip_build_cpp=skip_build_cpp,
    )
end

function parse_args()
    opts = Options()
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--pilots"
            i += 1
            i <= length(ARGS) || error("Missing value after --pilots")
            vals = [parse(Float64, strip(x)) for x in split(ARGS[i], ',') if !isempty(strip(x))]
            isempty(vals) && error("No pilot fractions parsed from --pilots")
            opts = replace_opts(opts; pilot_fracs=vals)
        elseif arg == "--bundle-dir"
            i += 1
            i <= length(ARGS) || error("Missing value after --bundle-dir")
            opts = replace_opts(opts; bundle_dir=ARGS[i])
        elseif arg == "--julia-csv"
            i += 1
            i <= length(ARGS) || error("Missing value after --julia-csv")
            opts = replace_opts(opts; julia_csv=ARGS[i])
        elseif arg == "--cpp-csv"
            i += 1
            i <= length(ARGS) || error("Missing value after --cpp-csv")
            opts = replace_opts(opts; cpp_csv=ARGS[i])
        elseif arg == "--merged-csv"
            i += 1
            i <= length(ARGS) || error("Missing value after --merged-csv")
            opts = replace_opts(opts; merged_csv=ARGS[i])
        elseif arg == "--summary-csv"
            i += 1
            i <= length(ARGS) || error("Missing value after --summary-csv")
            opts = replace_opts(opts; summary_csv=ARGS[i])
        elseif arg == "--skip-export"
            opts = replace_opts(opts; skip_export=true)
        elseif arg == "--skip-build-cpp"
            opts = replace_opts(opts; skip_build_cpp=true)
        elseif arg == "--help"
            println("""
            Usage:
              julia --project=. scripts/compare_julia_cpp_900.jl [options]

            Options:
              --pilots a,b,c       Compare only these pilot fractions
              --bundle-dir PATH    Bundle directory for the C++ runner
              --julia-csv PATH     Julia results CSV path
              --cpp-csv PATH       C++ results CSV path
              --merged-csv PATH    Merged comparison CSV path
              --summary-csv PATH   Summary CSV path
              --skip-export        Reuse an existing bundle instead of regenerating it
              --skip-build-cpp     Reuse an existing C++ binary instead of rebuilding
            """)
            exit(0)
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end
    return opts
end

function write_complex_matrix(path::String, mat)
    rows, cols = size(mat)
    open(path, "w") do io
        write(io, Int64(rows))
        write(io, Int64(cols))
        data = ComplexF64.(mat)
        for row in 1:rows, col in 1:cols
            value = data[row, col]
            write(io, Float64(real(value)))
            write(io, Float64(imag(value)))
        end
    end
end

function write_complex_vector(path::String, vec)
    write_complex_matrix(path, reshape(ComplexF64.(vec), 1, :))
end

function write_int_vector(path::String, vec)
    open(path, "w") do io
        write(io, Int64(length(vec)))
        for value in Int64.(vec)
            write(io, value)
        end
    end
end

function prepare_cache_from_config(project_root::String)
    data_dir = isabspath(DATA_DIR) ? DATA_DIR : joinpath(project_root, DATA_DIR)
    all_ytrain_df, all_packets_df, _ = ExperimentCore.load_data(data_dir, DATA_FILE, SIGNAL_FILE)
    num_frame_repeats = nrow(all_ytrain_df)
    x_train = ExperimentCore.prepare_training_signals(NUM_TRAIN)
    x_datas = ExperimentCore.prepare_data_signals(
        D_NODES, T_NODES, NPC, NUM_TRAIN, NUM_DATA, GAP, num_frame_repeats, data_dir
    )
    H_est_omp, H_est_mmse = ExperimentCore.estimate_channels(
        ExperimentCore.extract_signal_matrix(all_ytrain_df),
        x_train, H_LEN, K_SPARSE, NOISE_VARIANCE
    )
    cache = ExperimentCore.prepare_cache(all_ytrain_df, all_packets_df, x_train, x_datas, H_est_omp, H_est_mmse)
    return cache, data_dir
end

function export_cpp_bundle(bundle_dir::String, cache::ExperimentCore.DataCache, data_dir::String,
                           pilot_fracs::Vector{Float64})
    mkpath(bundle_dir)
    write_complex_matrix(joinpath(bundle_dir, "y_train_matrix.c64bin"), cache.y_train_matrix)
    write_complex_matrix(joinpath(bundle_dir, "packet_matrix.c64bin"), cache.packet_matrix)
    write_complex_vector(joinpath(bundle_dir, "x_train.c64bin"), cache.x_train)
    write_complex_matrix(joinpath(bundle_dir, "x_datas.c64bin"), cache.x_datas)
    write_int_vector(joinpath(bundle_dir, "y_train_frames.i64bin"), cache.y_train_frames)
    write_int_vector(joinpath(bundle_dir, "packet_frames.i64bin"), cache.packet_frames)
    write_int_vector(joinpath(bundle_dir, "packet_blocks.i64bin"), cache.packet_blocks)

    open(joinpath(bundle_dir, "manifest.txt"), "w") do io
        println(io, "data_dir=", data_dir)
        println(io, "ldpc_h_file=", joinpath(data_dir, LDPC_H_FILE))
        println(io, "num_frames_to_process=", size(cache.packet_matrix, 1))
        println(io, "packet_rows=", size(cache.packet_matrix, 1))
        println(io, "train_rows=", size(cache.y_train_matrix, 1))
        println(io, "pilot_fracs=", join(string.(pilot_fracs), ","))
        println(io, "h_len=", H_LEN)
        println(io, "k_sparse=", K_SPARSE)
        println(io, "noise_variance=", NOISE_VARIANCE)
        println(io, "d_nodes=", D_NODES)
        println(io, "t_nodes=", T_NODES)
        println(io, "npc=", NPC)
        println(io, "code_k=", CODE_K)
        println(io, "code_n=", CODE_N)
        println(io, "num_train=", NUM_TRAIN)
        println(io, "num_data=", NUM_DATA)
        println(io, "gap=", GAP)
        println(io, "lambda=", LAMBDA)
        println(io, "gamma=", GAMMA)
        println(io, "eta=", ETA)
        println(io, "results_file=ldpc_ber_cpp_900.csv")
    end
end

function run_julia_results(cache::ExperimentCore.DataCache, data_dir::String,
                           pilot_fracs::Vector{Float64})
    tr_len = length(cache.x_train)
    packet_rows = collect(1:size(cache.packet_matrix, 1))
    all_results = DataFrame(
        frame=Int[], block=Int[], ber_grad=Float64[], ber_dfe=Float64[],
        ber_spa=Float64[], ber_min=Float64[], pilot_frac=Float64[]
    )

    for pilot_frac in pilot_fracs
        @printf("Running Julia pipeline for pilot %.2f over %d packets...\n",
            pilot_frac, length(packet_rows)
        )
        res = redirect_stdout(devnull) do
            ExperimentCore.process_pilot_fraction(
                cache, pilot_frac, packet_rows,
                D_NODES, T_NODES, NPC, CODE_N, tr_len, H_LEN,
                LAMBDA, GAMMA, ETA, BPSK_CONSTELLATION, data_dir
            )
        end
        append!(all_results, res)
    end

    sort!(all_results, [:pilot_frac, :frame, :block])
    return all_results
end

function run_cpp_results(project_root::String, bundle_dir::String, cpp_csv::String, skip_build_cpp::Bool)
    cpp_dir = joinpath(project_root, "cpp")
    exe = joinpath(cpp_dir, "run_experiment_cpp")
    skip_build_cpp || run(`make -C $cpp_dir`)
    isfile(exe) || error("Missing C++ executable at $exe")
    println("Running C++ pipeline over exported bundle...")
    run(pipeline(`$exe $bundle_dir $cpp_csv`, stdout=devnull))
    df = CSV.read(cpp_csv, DataFrame)
    sort!(df, [:pilot_frac, :frame, :block])
    return df
end

function normalize_keys!(df::DataFrame)
    df[!, :frame] = Int.(df.frame)
    df[!, :block] = Int.(df.block)
    df[!, :pilot_frac] = round.(Float64.(df.pilot_frac), digits=8)
    return df
end

function compare_results(julia_df::DataFrame, cpp_df::DataFrame)
    normalize_keys!(julia_df)
    normalize_keys!(cpp_df)

    left = rename(copy(julia_df),
        :ber_grad => :julia_ber_grad,
        :ber_dfe => :julia_ber_dfe,
        :ber_spa => :julia_ber_spa,
        :ber_min => :julia_ber_min,
    )
    right = rename(copy(cpp_df),
        :ber_grad => :cpp_ber_grad,
        :ber_dfe => :cpp_ber_dfe,
        :ber_spa => :cpp_ber_spa,
        :ber_min => :cpp_ber_min,
    )

    merged = innerjoin(left, right, on=[:frame, :block, :pilot_frac])
    for metric in (:ber_grad, :ber_dfe, :ber_spa, :ber_min)
        lcol = Symbol("julia_", metric)
        rcol = Symbol("cpp_", metric)
        merged[!, Symbol("signeddiff_", metric)] = merged[!, lcol] .- merged[!, rcol]
        merged[!, Symbol("absdiff_", metric)] = abs.(merged[!, Symbol("signeddiff_", metric)])
    end
    sort!(merged, [:pilot_frac, :frame, :block])
    return merged
end

function summarize_results(merged::DataFrame)
    groups = groupby(merged, :pilot_frac)
    summary = DataFrame(
        pilot_frac=Float64[],
        packets=Int[],
        julia_ber_grad_mean=Float64[],
        cpp_ber_grad_mean=Float64[],
        julia_ber_dfe_mean=Float64[],
        cpp_ber_dfe_mean=Float64[],
        julia_ber_spa_mean=Float64[],
        cpp_ber_spa_mean=Float64[],
        julia_ber_min_mean=Float64[],
        cpp_ber_min_mean=Float64[],
        mean_absdiff_ber_grad=Float64[],
        max_absdiff_ber_grad=Float64[],
    )

    for group in groups
        push!(summary, (
            pilot_frac=first(group.pilot_frac),
            packets=nrow(group),
            julia_ber_grad_mean=mean(group.julia_ber_grad),
            cpp_ber_grad_mean=mean(group.cpp_ber_grad),
            julia_ber_dfe_mean=mean(group.julia_ber_dfe),
            cpp_ber_dfe_mean=mean(group.cpp_ber_dfe),
            julia_ber_spa_mean=mean(group.julia_ber_spa),
            cpp_ber_spa_mean=mean(group.cpp_ber_spa),
            julia_ber_min_mean=mean(group.julia_ber_min),
            cpp_ber_min_mean=mean(group.cpp_ber_min),
            mean_absdiff_ber_grad=mean(group.absdiff_ber_grad),
            max_absdiff_ber_grad=maximum(group.absdiff_ber_grad),
        ))
    end

    sort!(summary, :pilot_frac)
    return summary
end

function print_summary(summary::DataFrame, merged::DataFrame)
    println()
    println("="^118)
    println("Julia vs C++ BER Summary over 900 packets")
    println("="^118)
    println("pilot   n     julia_dfec  cpp_dfec    julia_dfe   cpp_dfe     julia_spa   cpp_spa     julia_min   cpp_min")
    println("-"^118)
    for row in eachrow(summary)
        @printf("%.2f   %4d   %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f\n",
            row.pilot_frac, row.packets,
            row.julia_ber_grad_mean, row.cpp_ber_grad_mean,
            row.julia_ber_dfe_mean, row.cpp_ber_dfe_mean,
            row.julia_ber_spa_mean, row.cpp_ber_spa_mean,
            row.julia_ber_min_mean, row.cpp_ber_min_mean
        )
    end
    println("-"^118)
    @printf("overall %4d   %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f  %10.6f\n",
        nrow(merged),
        mean(merged.julia_ber_grad), mean(merged.cpp_ber_grad),
        mean(merged.julia_ber_dfe), mean(merged.cpp_ber_dfe),
        mean(merged.julia_ber_spa), mean(merged.cpp_ber_spa),
        mean(merged.julia_ber_min), mean(merged.cpp_ber_min)
    )
    println()
    println("DFEC difference summary:")
    for row in eachrow(summary)
        @printf("pilot %.2f -> mean |Julia-C++| DFEC = %.6e, max |Julia-C++| DFEC = %.6e\n",
            row.pilot_frac, row.mean_absdiff_ber_grad, row.max_absdiff_ber_grad
        )
    end
end

function main()
    opts = parse_args()
    project_root = dirname(@__DIR__)
    results_dir = joinpath(project_root, "results")
    bundle_dir = isempty(opts.bundle_dir) ? joinpath(project_root, "export", "cpp_bundle") : abspath(opts.bundle_dir)
    julia_csv = isempty(opts.julia_csv) ? joinpath(results_dir, "ldpc_ber_julia_900.csv") : abspath(opts.julia_csv)
    cpp_csv = isempty(opts.cpp_csv) ? joinpath(results_dir, "ldpc_ber_cpp_900.csv") : abspath(opts.cpp_csv)
    merged_csv = isempty(opts.merged_csv) ? joinpath(results_dir, "compare_julia_cpp_900.csv") : abspath(opts.merged_csv)
    summary_csv = isempty(opts.summary_csv) ? joinpath(results_dir, "compare_julia_cpp_900_summary.csv") : abspath(opts.summary_csv)

    mkpath(results_dir)
    cache, data_dir = prepare_cache_from_config(project_root)

    if opts.skip_export
        println("Reusing existing C++ bundle at: ", bundle_dir)
    else
        println("Exporting C++ bundle to: ", bundle_dir)
        export_cpp_bundle(bundle_dir, cache, data_dir, opts.pilot_fracs)
    end

    julia_df = run_julia_results(cache, data_dir, opts.pilot_fracs)
    CSV.write(julia_csv, julia_df)
    println("Saved Julia CSV to: ", julia_csv)

    cpp_df = run_cpp_results(project_root, bundle_dir, cpp_csv, opts.skip_build_cpp)
    println("Saved C++ CSV to: ", cpp_csv)

    merged = compare_results(julia_df, cpp_df)
    CSV.write(merged_csv, merged)
    println("Saved merged CSV to: ", merged_csv)

    summary = summarize_results(merged)
    CSV.write(summary_csv, summary)
    println("Saved summary CSV to: ", summary_csv)

    print_summary(summary, merged)
end

main()
