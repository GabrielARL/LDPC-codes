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
using DataFrames
using LinearAlgebra
using Optim
using Printf

Base.@kwdef struct Options
    packet_index::Int = 1
    pilot_frac::Float64 = first(Float64.(collect(PILOT_FRACTIONS)))
end

function parse_args()
    opts = Options()
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--packet"
            i += 1
            i <= length(ARGS) || error("Missing value after --packet")
            opts = Options(packet_index=parse(Int, ARGS[i]), pilot_frac=opts.pilot_frac)
        elseif arg == "--pilot"
            i += 1
            i <= length(ARGS) || error("Missing value after --pilot")
            opts = Options(packet_index=opts.packet_index, pilot_frac=parse(Float64, ARGS[i]))
        elseif arg == "--help"
            println("""
            Usage:
              julia --project=. scripts/debug_julia_cpp_gap.jl [options]

            Options:
              --packet N     Packet row to analyze (default: 1)
              --pilot P      Pilot fraction to analyze (default: $(opts.pilot_frac))
            """)
            exit(0)
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end
    return opts
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

function cpp_style_dfe_equalize(y::AbstractVector, taps::AbstractVector)
    taps_c = ComplexF64.(taps)
    y_c = ComplexF64.(y)
    iszero(taps_c[1]) && error("cpp_style_dfe_equalize requires non-zero main tap")
    out = zeros(ComplexF64, length(y_c))
    decisions = zeros(Float64, length(y_c))
    for i in eachindex(y_c)
        fb = 0.0 + 0.0im
        for tap in 2:length(taps_c)
            prev_idx = i - tap + 1
            if prev_idx >= 1
                fb += taps_c[tap] * decisions[prev_idx]
            end
        end
        eq = (y_c[i] - fb) / taps_c[1]
        out[i] = eq
        decisions[i] = real(eq) >= 0 ? 1.0 : -1.0
    end
    return out
end

function cpp_style_estimate_omp_channel(y_train::AbstractVector, x_train::AbstractVector, h_len::Int, k_sparse::Int)
    y_c = ComplexF64.(y_train)
    x_c = ComplexF64.(x_train)
    n = length(y_c)
    X = zeros(ComplexF64, n, h_len)
    for i in 1:n, j in 1:h_len
        if i - j + 1 >= 1
            X[i, j] = x_c[i - j + 1]
        end
    end

    function solve_support(active::Vector{Int})
        Xs = X[:, active]
        gram = Xs' * Xs
        for i in 1:size(gram, 1)
            gram[i, i] += 1e-8
        end
        rhs = Xs' * y_c
        return gram \ rhs
    end

    residual = copy(y_c)
    support = Int[]
    used = falses(h_len)
    for _ in 1:k_sparse
        correlations = fill(-Inf, h_len)
        for j in 1:h_len
            if !used[j]
                correlations[j] = abs(sum(conj.(X[:, j]) .* residual))
            end
        end
        best_j = argmax(correlations)
        push!(support, best_j)
        used[best_j] = true
        coeffs = solve_support(support)
        residual = y_c - X[:, support] * coeffs
    end

    h = zeros(ComplexF64, h_len)
    coeffs = solve_support(support)
    h[support] = coeffs
    return h
end

function parse_h_rows_0based(path::String)
    rows = Vector{Vector{Int}}()
    for line in eachline(path)
        m = match(r"^\s*(\d+):(.+)$", line)
        m === nothing && continue
        row = parse(Int, m[1]) + 1
        while length(rows) < row
            push!(rows, Int[])
        end
        rows[row] = [parse(Int, tok) for tok in split(strip(m[2]))]
    end
    return rows
end

function cpp_style_pilot_positions(h_path::String, pilot_frac::Float64)
    rows0 = parse_h_rows_0based(h_path)
    num_rows = length(rows0)
    start_row = round(Int, (1.0 - pilot_frac) * num_rows) + 1
    start_row = clamp(start_row, 1, num_rows)
    pilots0 = sort(unique(vcat(rows0[start_row:end]...)))
    return pilots0 .+ 1
end

function symbol_ber(truth_symbols::AbstractVector, est_symbols::AbstractVector,
                    nonpilot::Vector{Int}, denom_n::Int)
    errors = sum(sign.(real.(truth_symbols[nonpilot])) .!= sign.(real.(est_symbols[nonpilot])))
    return errors / denom_n
end

function summarize_seed(label::String, x_seed, y_data, code, idrows, pilot, pilot_bpsk,
                        x_pilot, x_ref, nonpilot, h_pos, h_init)
    z_init = ExperimentCore.warm_start_logits(x_seed, pilot, x_pilot)
    _, _, result = LDPCJDPMemoized.decode_sparse_joint(
        y_data, code, idrows, pilot, pilot_bpsk, h_pos;
        h_init=h_init, z_init=z_init, λ=LAMBDA, γ=GAMMA, η=ETA, verbose=false
    )
    n_code = code.n
    z_opt = Optim.minimizer(result)[1:n_code]
    x_soft = tanh.(z_opt)
    _, phase = SignalUtils.argminphase(x_soft[pilot], x_pilot)
    x_rot = x_soft .* exp(-im * deg2rad(-phase))
    ber = symbol_ber(x_ref, x_rot, nonpilot, n_code)
    return (
        label=label,
        ber=ber,
        valid=result.valid,
        objective=result.objective,
        z_rms=sqrt(sum(abs2, z_init) / length(z_init)),
    )
end

function main()
    opts = parse_args()
    project_root = dirname(@__DIR__)
    cache, data_dir = prepare_cache_from_config(project_root)
    packet_count = size(cache.packet_matrix, 1)
    1 <= opts.packet_index <= packet_count || error("Packet index $(opts.packet_index) out of bounds for $packet_count packets")

    code = cols = idrows = pilot = nothing
    old_dir = pwd()
    try
        cd(data_dir)
        code, cols, idrows, pilot = LDPCJDPMemoized.initcode(
            D_NODES, T_NODES, NPC; pilot_row_fraction=opts.pilot_frac
        )
    finally
        cd(old_dir)
    end

    n_code = code.n
    nonpilot = sort(setdiff(1:n_code, pilot))
    pilot_cpp = cpp_style_pilot_positions(joinpath(data_dir, LDPC_H_FILE), opts.pilot_frac)
    nonpilot_cpp = sort(setdiff(1:n_code, pilot_cpp))
    train_idx = cache.packet_to_train_idx[opts.packet_index]
    xdata_idx = cache.packet_to_xdata_idx[opts.packet_index]
    frame_id = cache.packet_frames[opts.packet_index]
    block_id = cache.packet_blocks[opts.packet_index]

    y_data = ComplexF64.(cache.packet_matrix[opts.packet_index, 1:n_code])
    x_ref = ComplexF64.(cache.x_datas[xdata_idx, 1:n_code])
    x_pilot = x_ref[pilot]
    pilot_data = LDPCJDPMemoized.demodulate.(x_pilot)
    pilot_bpsk = LDPCJDPMemoized.modulate.(pilot_data)
    y_train = ComplexF64.(cache.y_train_matrix[train_idx, 1:length(cache.x_train)])
    h_julia = ComplexF64.(cache.H_est_omp[train_idx, :])
    h_cpp = cpp_style_estimate_omp_channel(y_train, cache.x_train, H_LEN, K_SPARSE)
    h_pos = findall(!iszero, h_julia)
    h_init = h_julia[h_pos]
    h_pos_cpp = findall(!iszero, h_cpp)
    h_init_cpp = h_cpp[h_pos_cpp]

    tr_len = length(cache.x_train)
    x_julia = ExperimentCore.dfe_equalize_packet(
        cache, train_idx, xdata_idx, y_data, tr_len, H_LEN, BPSK_CONSTELLATION, length(pilot)
    )
    x_cpp_raw = cpp_style_dfe_equalize(y_data, h_julia)
    x_cpp_aligned, cpp_phase = SignalUtils.argminphase(x_cpp_raw, x_ref)
    x_cpp_omp_raw = cpp_style_dfe_equalize(y_data, h_cpp)
    x_cpp_omp_aligned, cpp_omp_phase = SignalUtils.argminphase(x_cpp_omp_raw, x_ref)

    ber_julia = symbol_ber(x_ref, x_julia, nonpilot, n_code)
    ber_cpp_raw = symbol_ber(x_ref, x_cpp_raw, nonpilot, n_code)
    ber_cpp_aligned = symbol_ber(x_ref, x_cpp_aligned, nonpilot, n_code)
    ber_cpp_omp_raw = symbol_ber(x_ref, x_cpp_omp_raw, nonpilot, n_code)
    ber_cpp_omp_aligned = symbol_ber(x_ref, x_cpp_omp_aligned, nonpilot, n_code)
    ber_cpp_raw_cpppilot = symbol_ber(x_ref, x_cpp_raw, nonpilot_cpp, n_code)

    julia_joint = summarize_seed(
        "julia adaptive dfe", x_julia, y_data, code, idrows, pilot, pilot_bpsk,
        x_pilot, x_ref, nonpilot, h_pos, h_init
    )
    cpp_raw_joint = summarize_seed(
        "cpp simple dfe raw", x_cpp_raw, y_data, code, idrows, pilot, pilot_bpsk,
        x_pilot, x_ref, nonpilot, h_pos, h_init
    )
    cpp_aligned_joint = summarize_seed(
        "cpp simple dfe + phase", x_cpp_aligned, y_data, code, idrows, pilot, pilot_bpsk,
        x_pilot, x_ref, nonpilot, h_pos, h_init
    )
    cpp_omp_joint = summarize_seed(
        "cpp omp + simple dfe", x_cpp_omp_raw, y_data, code, idrows, pilot, pilot_bpsk,
        x_pilot, x_ref, nonpilot, h_pos_cpp, h_init_cpp
    )

    println("="^92)
    @printf("Debug Julia vs C++ gap for packet row %d (frame %d, block %d), pilot %.2f\n",
        opts.packet_index, frame_id, block_id, opts.pilot_frac
    )
    println("="^92)
    @printf("Training row: %d  xdata row: %d  pilot count: %d  nonpilot count: %d\n",
        train_idx, xdata_idx, length(pilot), length(nonpilot)
    )
    @printf("Pilot-set mismatch count    : %d\n", length(symdiff(Set(pilot), Set(pilot_cpp))))
    @printf("Julia OMP support taps: %d  main tap magnitude: %.6f\n", length(h_pos), abs(h_julia[1]))
    @printf("C++   OMP support taps: %d  main tap magnitude: %.6f\n", length(h_pos_cpp), abs(h_cpp[1]))
    @printf("OMP tap L2 difference   : %.6e\n", norm(h_julia - h_cpp))
    println()
    println("DFE BER on the same packet and same OMP channel:")
    @printf("  Julia adaptive DFE           : %.6f\n", ber_julia)
    @printf("  C++ simple DFE raw           : %.6f\n", ber_cpp_raw)
    @printf("  C++ simple DFE raw (C++ pilots): %.6f\n", ber_cpp_raw_cpppilot)
    @printf("  C++ simple DFE + phase align : %.6f  (phase %.1f deg)\n", ber_cpp_aligned, cpp_phase)
    @printf("  C++ OMP + simple DFE raw     : %.6f\n", ber_cpp_omp_raw)
    @printf("  C++ OMP + simple DFE + phase : %.6f  (phase %.1f deg)\n", ber_cpp_omp_aligned, cpp_omp_phase)
    println()
    println("Julia joint decoder re-run with different warm starts:")
    for item in (julia_joint, cpp_raw_joint, cpp_aligned_joint, cpp_omp_joint)
        @printf("  %-24s BER %.6f  valid=%s  objective=%.6f  z_rms=%.4f\n",
            item.label, item.ber, item.valid ? "yes" : "no", item.objective, item.z_rms
        )
    end
    println()
    println("Notes:")
    println("  1. The first two rows isolate the equalizer difference only, using the same Julia OMP channel.")
    println("  2. The joint-decoder rows keep Julia's decoder fixed and only swap the warm start.")
    println("  3. If the raw C++ warm start is much worse but the phase-aligned one recovers, the missing phase alignment is a major source of the gap.")
    println("  4. Any remaining gap after phase alignment points at the equalizer structure and/or the separate C++ joint optimizer.")
end

main()
