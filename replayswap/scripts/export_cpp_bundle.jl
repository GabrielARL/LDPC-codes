#!/usr/bin/env julia

using Random
using JLD2
using DataFrames
using SignalAnalysis

include(joinpath(@__DIR__, "..", "lib", "paths.jl"))
include(joinpath(ROOT, "lib", "LDPCJDPMemoized.jl"))
import .LDPCJDPMemoized as LDM

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

function write_int_matrix(path::String, mat)
    rows, cols = size(mat)
    open(path, "w") do io
        write(io, Int64(rows))
        write(io, Int64(cols))
        data = Int64.(mat)
        for row in 1:rows, col in 1:cols
            write(io, data[row, col])
        end
    end
end

function write_int_vector(path::String, vec)
    open(path, "w") do io
        write(io, Int64(length(vec)))
        for value in Int64.(vec)
            write(io, value)
        end
    end
end

function write_float_vector(path::String, vec)
    open(path, "w") do io
        write(io, Int64(length(vec)))
        for value in Float64.(vec)
            write(io, value)
        end
    end
end

function cw_true_matrix(num_packets::Int)
    codeB, _colsB, _idrowsB, _ = LDM.initcode(64, 128, 4; pilot_row_fraction=0.10)
    m11 = mseq(11)
    out = Matrix{Int64}(undef, num_packets, codeB.n)
    for row in 1:num_packets
        idx = ((row - 1) % 20) + 1
        bseq = m11[idx : (codeB.k + idx - 1)]
        d_test = Int.((bseq .+ 1) ./ 2)
        out[row, :] = Int64.(LDM.encode(codeB, d_test))
    end
    return out
end

function main()
    out_dir = if length(ARGS) >= 1
        abspath(ARGS[1])
    else
        normpath(joinpath(ROOT, "..", "replayswap_cpp", "export", "default_bundle"))
    end
    mkpath(out_dir)

    raw_path = joinpath(DATA_DIR, "raw", "logged_packets_and_ytrain.jld2")
    cache_path = joinpath(DATA_DIR, "ls_cache_h20_rho1e-02_bestD.jld2")
    rsc_path = joinpath(DATA_DIR, "replayswap_bpsk_RSC_64_128_from_realdata_donorLS_h20_rho1e-2.jld2")
    ldpc_h_src = joinpath(CODES_DIR, "ldpc", "64-128-4.H")

    isfile(raw_path) || error("Missing RAW dataset: $raw_path")
    isfile(cache_path) || error("Missing LS cache: $cache_path")
    isfile(rsc_path) || error("Missing RSC dataset: $rsc_path")
    isfile(ldpc_h_src) || error("Missing LDPC H file: $ldpc_h_src")

    raw_data = JLD2.load(raw_path)
    all_packets_df = DataFrame(raw_data["all_packets_df"])
    packet_cols = Symbol.("s" .* string.(1:128))
    packet_matrix = ComplexF64.(Matrix(select(all_packets_df, packet_cols)))
    raw_frame_ids = Int64.(all_packets_df.frame)
    raw_block_ids = hasproperty(all_packets_df, :block) ? Int64.(all_packets_df.block) : ones(Int64, nrow(all_packets_df))

    cache = JLD2.load(cache_path)
    raw_bestD = Int64.(cache["bestD"])
    length(raw_bestD) == nrow(all_packets_df) || error("bestD length mismatch")

    raw_cw_true = cw_true_matrix(nrow(all_packets_df))

    rsc = JLD2.load(rsc_path)
    rsc_y = ComplexF64.(rsc["y_bpsk_swapped"])
    rsc_u64 = Int64.(rsc["u64_mat"])
    rsc_b128 = Int64.(rsc["b128_mat"])
    rsc_h = ComplexF64.(rsc["h_blk_mat"])
    rsc_corr = Float64.(rsc["corr_donor"])
    rsc_frame_ids = Int64.(vec(rsc["frame_ids"]))
    eligible = findall(rsc_corr .>= 0.10)
    shuffle!(MersenneTwister(12648430), eligible)
    rsc_default_order = Int64.(eligible .- 1)

    write_complex_matrix(joinpath(out_dir, "raw_packets.c64bin"), packet_matrix)
    write_int_vector(joinpath(out_dir, "raw_frame_ids.i64bin"), raw_frame_ids)
    write_int_vector(joinpath(out_dir, "raw_block_ids.i64bin"), raw_block_ids)
    write_int_vector(joinpath(out_dir, "raw_bestD.i64bin"), raw_bestD)
    write_int_matrix(joinpath(out_dir, "raw_cw_true.i64bin"), raw_cw_true)

    write_complex_matrix(joinpath(out_dir, "rsc_y.c64bin"), rsc_y)
    write_int_matrix(joinpath(out_dir, "rsc_u64.i64bin"), rsc_u64)
    write_int_matrix(joinpath(out_dir, "rsc_b128.i64bin"), rsc_b128)
    write_complex_matrix(joinpath(out_dir, "rsc_h.c64bin"), rsc_h)
    write_float_vector(joinpath(out_dir, "rsc_corr.f64bin"), rsc_corr)
    write_int_vector(joinpath(out_dir, "rsc_frame_ids.i64bin"), rsc_frame_ids)
    write_int_vector(joinpath(out_dir, "rsc_default_order.i64bin"), rsc_default_order)

    cp(ldpc_h_src, joinpath(out_dir, basename(ldpc_h_src)); force=true)

    open(joinpath(out_dir, "manifest.txt"), "w") do io
        println(io, "bundle_version=1")
        println(io, "bundle_name=replayswap_bpsk_1_2")
        println(io, "ldpc_h_file=64-128-4.H")
        println(io, "raw_packets_file=raw_packets.c64bin")
        println(io, "raw_frame_ids_file=raw_frame_ids.i64bin")
        println(io, "raw_block_ids_file=raw_block_ids.i64bin")
        println(io, "raw_bestD_file=raw_bestD.i64bin")
        println(io, "raw_cw_true_file=raw_cw_true.i64bin")
        println(io, "rsc_y_file=rsc_y.c64bin")
        println(io, "rsc_u64_file=rsc_u64.i64bin")
        println(io, "rsc_b128_file=rsc_b128.i64bin")
        println(io, "rsc_h_file=rsc_h.c64bin")
        println(io, "rsc_corr_file=rsc_corr.f64bin")
        println(io, "rsc_frame_ids_file=rsc_frame_ids.i64bin")
        println(io, "rsc_default_order_file=rsc_default_order.i64bin")
        println(io, "pilot_fracs=0.0,0.1,0.2,0.3,0.4,0.5")
        println(io, "raw_start_frame=1")
        println(io, "raw_n_per_p=20")
        println(io, "raw_h_len=20")
        println(io, "raw_rho_ls=0.01")
        println(io, "raw_lambda=2.0")
        println(io, "raw_lambda_pil=20.0")
        println(io, "raw_gamma=0.001")
        println(io, "raw_eta=1.0")
        println(io, "raw_k_sparse=4")
        println(io, "raw_max_iter_opt=20")
        println(io, "rsc_corr_thr=0.10")
        println(io, "rsc_nblk=200")
        println(io, "rsc_seed_sel=12648430")
        println(io, "rsc_start=1")
        println(io, "rsc_turbo_iters=2")
        println(io, "rsc_sigma2_init=1.30")
        println(io, "rsc_eq_sigma2_iters=1")
        println(io, "rsc_llr_clip=25.0")
        println(io, "rsc_default_order_corr_thr=0.10")
        println(io, "rsc_default_order_seed=12648430")
    end

    println("Exported replayswap C++ bundle to: $out_dir")
    println("  RAW packets: $(size(packet_matrix, 1)) x $(size(packet_matrix, 2))")
    println("  RAW truths:  $(size(raw_cw_true, 1)) x $(size(raw_cw_true, 2))")
    println("  RSC y:       $(size(rsc_y, 1)) x $(size(rsc_y, 2))")
    println("  RSC u64:     $(size(rsc_u64, 1)) x $(size(rsc_u64, 2))")
    println("  RSC b128:    $(size(rsc_b128, 1)) x $(size(rsc_b128, 2))")
end

main()
