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
using JLD2
using DataFrames

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

function main()
    project_root = dirname(@__DIR__)
    out_dir = length(ARGS) >= 1 ? abspath(ARGS[1]) : joinpath(project_root, "export", "cpp_bundle")
    mkpath(out_dir)

    data_dir = isabspath(DATA_DIR) ? DATA_DIR : joinpath(project_root, DATA_DIR)

    all_ytrain_df, all_packets_df, _ = ExperimentCore.load_data(data_dir, DATA_FILE, SIGNAL_FILE)
    num_frame_repeats = nrow(all_ytrain_df)
    x_train = ExperimentCore.prepare_training_signals(NUM_TRAIN)
    x_datas = ExperimentCore.prepare_data_signals(
        D_NODES, T_NODES, NPC, NUM_TRAIN, NUM_DATA, GAP, num_frame_repeats, data_dir
    )

    y_train_matrix = ExperimentCore.extract_signal_matrix(all_ytrain_df)
    packet_matrix = ExperimentCore.extract_signal_matrix(all_packets_df)
    y_train_frames, packet_frames, packet_blocks =
        ExperimentCore.extract_row_metadata(all_ytrain_df)[1],
        ExperimentCore.extract_row_metadata(all_packets_df)[1],
        ExperimentCore.extract_row_metadata(all_packets_df)[2]

    write_complex_matrix(joinpath(out_dir, "y_train_matrix.c64bin"), y_train_matrix)
    write_complex_matrix(joinpath(out_dir, "packet_matrix.c64bin"), packet_matrix)
    write_complex_vector(joinpath(out_dir, "x_train.c64bin"), x_train)
    write_complex_matrix(joinpath(out_dir, "x_datas.c64bin"), x_datas)
    write_int_vector(joinpath(out_dir, "y_train_frames.i64bin"), y_train_frames)
    write_int_vector(joinpath(out_dir, "packet_frames.i64bin"), packet_frames)
    write_int_vector(joinpath(out_dir, "packet_blocks.i64bin"), packet_blocks)

    open(joinpath(out_dir, "manifest.txt"), "w") do io
        println(io, "data_dir=", data_dir)
        println(io, "ldpc_h_file=", joinpath(data_dir, LDPC_H_FILE))
        println(io, "num_frames_to_process=", size(packet_matrix, 1))
        println(io, "packet_rows=", size(packet_matrix, 1))
        println(io, "train_rows=", size(y_train_matrix, 1))
        println(io, "pilot_fracs=", join(string.(collect(PILOT_FRACTIONS)), ","))
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
        println(io, "results_file=", RESULTS_FILE)
    end

    println("Exported C++ DFEC bundle to: ", out_dir)
    println("  y_train rows: ", size(y_train_matrix, 1), "  packet rows: ", size(packet_matrix, 1))
    println("  x_datas rows: ", size(x_datas, 1), "  code length: ", size(x_datas, 2))
end

main()
