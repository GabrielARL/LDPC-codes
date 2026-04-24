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

Base.@kwdef struct Options
    packet_index::Int = 1
    pilot_fracs::Vector{Float64} = Float64.(collect(PILOT_FRACTIONS))
    output_csv::String = ""
    skip_build_cpp::Bool = false
    keep_bundle::Bool = false
    bundle_dir::String = ""
end

function replace_opts(opts::Options; packet_index::Int=opts.packet_index,
                      pilot_fracs::Vector{Float64}=opts.pilot_fracs,
                      output_csv::String=opts.output_csv,
                      skip_build_cpp::Bool=opts.skip_build_cpp,
                      keep_bundle::Bool=opts.keep_bundle,
                      bundle_dir::String=opts.bundle_dir)
    return Options(
        packet_index=packet_index,
        pilot_fracs=pilot_fracs,
        output_csv=output_csv,
        skip_build_cpp=skip_build_cpp,
        keep_bundle=keep_bundle,
        bundle_dir=bundle_dir,
    )
end

function parse_args()
    opts = Options()
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--packet"
            i += 1
            i <= length(ARGS) || error("Missing value after --packet")
            opts = replace_opts(opts; packet_index=parse(Int, ARGS[i]))
        elseif arg == "--pilots"
            i += 1
            i <= length(ARGS) || error("Missing value after --pilots")
            vals = [parse(Float64, strip(x)) for x in split(ARGS[i], ',') if !isempty(strip(x))]
            isempty(vals) && error("No pilot fractions parsed from --pilots")
            opts = replace_opts(opts; pilot_fracs=vals)
        elseif arg == "--out"
            i += 1
            i <= length(ARGS) || error("Missing value after --out")
            opts = replace_opts(opts; output_csv=ARGS[i])
        elseif arg == "--skip-build-cpp"
            opts = replace_opts(opts; skip_build_cpp=true)
        elseif arg == "--keep-bundle"
            opts = replace_opts(opts; keep_bundle=true)
        elseif arg == "--bundle-dir"
            i += 1
            i <= length(ARGS) || error("Missing value after --bundle-dir")
            opts = replace_opts(opts; keep_bundle=true, bundle_dir=ARGS[i])
        elseif arg == "--help"
            println("""
            Usage:
              julia --project=. scripts/compare_julia_cpp_packet.jl [options]

            Options:
              --packet N           Compare packet row N (default: 1)
              --pilots a,b,c       Compare only these pilot fractions
              --out PATH           Write merged comparison CSV to PATH
              --skip-build-cpp     Do not run `make` in cpp/
              --keep-bundle        Keep the temporary C++ bundle directory
              --bundle-dir PATH    Keep the generated bundle at PATH
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

function write_manifest(path::String, data_dir::String, pilot_fracs::Vector{Float64})
    open(path, "w") do io
        println(io, "data_dir=", data_dir)
        println(io, "ldpc_h_file=", joinpath(data_dir, LDPC_H_FILE))
        println(io, "num_frames_to_process=1")
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
        println(io, "results_file=ldpc_ber_cpp_packet.csv")
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

function build_julia_results(cache::ExperimentCore.DataCache, data_dir::String, packet_index::Int,
                             pilot_fracs::Vector{Float64})
    tr_len = length(cache.x_train)
    all_results = DataFrame(
        frame=Int[], block=Int[], ber_grad=Float64[], ber_dfe=Float64[],
        ber_spa=Float64[], ber_min=Float64[], pilot_frac=Float64[]
    )

    for pilot_frac in pilot_fracs
        res = ExperimentCore.process_pilot_fraction(
            cache, pilot_frac, [packet_index],
            D_NODES, T_NODES, NPC, CODE_N, tr_len, H_LEN,
            LAMBDA, GAMMA, ETA, BPSK_CONSTELLATION, data_dir
        )
        append!(all_results, res)
    end

    sort!(all_results, :pilot_frac)
    return all_results
end

function write_cpp_bundle(bundle_dir::String, cache::ExperimentCore.DataCache, data_dir::String,
                          packet_index::Int, pilot_fracs::Vector{Float64})
    mkpath(bundle_dir)
    write_complex_matrix(joinpath(bundle_dir, "y_train_matrix.c64bin"), cache.y_train_matrix)
    write_complex_matrix(
        joinpath(bundle_dir, "packet_matrix.c64bin"),
        reshape(cache.packet_matrix[packet_index, :], 1, size(cache.packet_matrix, 2))
    )
    write_complex_vector(joinpath(bundle_dir, "x_train.c64bin"), cache.x_train)
    write_complex_matrix(joinpath(bundle_dir, "x_datas.c64bin"), cache.x_datas)
    write_int_vector(joinpath(bundle_dir, "y_train_frames.i64bin"), cache.y_train_frames)
    write_int_vector(joinpath(bundle_dir, "packet_frames.i64bin"), [cache.packet_frames[packet_index]])
    write_int_vector(joinpath(bundle_dir, "packet_blocks.i64bin"), [cache.packet_blocks[packet_index]])
    write_manifest(joinpath(bundle_dir, "manifest.txt"), data_dir, pilot_fracs)
end

function run_cpp_results(project_root::String, bundle_dir::String, opts::Options)
    cpp_dir = joinpath(project_root, "cpp")
    exe = joinpath(cpp_dir, "run_experiment_cpp")
    opts.skip_build_cpp || run(`make -C $cpp_dir`)
    isfile(exe) || error("Missing C++ executable at $exe")

    output_csv = joinpath(bundle_dir, "ldpc_ber_cpp_packet.csv")
    run(`$exe $bundle_dir $output_csv`)
    df = CSV.read(output_csv, DataFrame)
    sort!(df, :pilot_frac)
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

    sort!(merged, :pilot_frac)
    return merged
end

function print_summary(merged::DataFrame, packet_index::Int)
    println()
    println("="^88)
    @printf("Julia vs C++ comparison for packet row %d\n", packet_index)
    println("="^88)
    println("pilot    frame  block  julia_dfec  cpp_dfec   |Δ|dfec   julia_dfe   cpp_dfe    |Δ|dfe")
    println("-"^88)
    for row in eachrow(merged)
        @printf("%.2f     %4d   %4d   %9.6f  %9.6f  %9.6f  %9.6f  %9.6f  %9.6f\n",
            row.pilot_frac, row.frame, row.block,
            row.julia_ber_grad, row.cpp_ber_grad, row.absdiff_ber_grad,
            row.julia_ber_dfe, row.cpp_ber_dfe, row.absdiff_ber_dfe
        )
    end
    println("-"^88)
    for metric in (:ber_grad, :ber_dfe, :ber_spa, :ber_min)
        col = Symbol("absdiff_", metric)
        @printf("max |Δ| %-7s = %.6e\n", String(metric), maximum(merged[!, col]))
    end
end

function main()
    opts = parse_args()
    project_root = dirname(@__DIR__)
    default_out = joinpath(
        project_root, "results", @sprintf("compare_julia_cpp_packet_%03d.csv", opts.packet_index)
    )
    output_csv = isempty(opts.output_csv) ? default_out : abspath(opts.output_csv)

    cache, data_dir = prepare_cache_from_config(project_root)
    1 <= opts.packet_index <= size(cache.packet_matrix, 1) ||
        error("Packet index $(opts.packet_index) is out of bounds for $(size(cache.packet_matrix, 1)) packets")

    julia_df = build_julia_results(cache, data_dir, opts.packet_index, opts.pilot_fracs)

    bundle_dir = if opts.keep_bundle
        isempty(opts.bundle_dir) ?
            joinpath(project_root, "export", @sprintf("compare_packet_%03d_bundle", opts.packet_index)) :
            abspath(opts.bundle_dir)
    else
        mktempdir()
    end

    try
        write_cpp_bundle(bundle_dir, cache, data_dir, opts.packet_index, opts.pilot_fracs)
        cpp_df = run_cpp_results(project_root, bundle_dir, opts)
        merged = compare_results(julia_df, cpp_df)

        mkpath(dirname(output_csv))
        CSV.write(output_csv, merged)
        print_summary(merged, opts.packet_index)
        println()
        println("Saved merged comparison CSV to: ", output_csv)
        if opts.keep_bundle
            println("Kept C++ bundle at: ", bundle_dir)
        end
    finally
        if !opts.keep_bundle && isdir(bundle_dir)
            rm(bundle_dir; recursive=true, force=true)
        end
    end
end

main()
