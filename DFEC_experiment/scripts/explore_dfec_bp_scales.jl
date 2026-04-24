"""
    explore_dfec_bp_scales.jl

Evaluate a few `DFEC -> BPdecoder` post-processing strategies on the
phase-corrected dataset to see which one improves BER over the current
direct hard decision on the joint decoder output.

Defaults:
  - pilot fraction: 0.41
  - packet rows:    1:50

Usage:

    julia --project=. scripts/explore_dfec_bp_scales.jl

Optional row range:

    julia --project=. scripts/explore_dfec_bp_scales.jl 1 100
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "config"))

include("../config/experiment_config.jl")
include("../src/FEC.jl")
include("../src/BPdecoder.jl")
include("../src/SignalUtils.jl")
include("../src/LDPCJDPMemoized.jl")
include("../src/ExperimentCore.jl")

using .FEC
using .BPdecoder
using .SignalUtils
using .LDPCJDPMemoized
using .ExperimentCore

using JLD2, DataFrames, Statistics, Optim, AdaptiveEstimators, SignalAnalysis, Printf

const PHASE_CORRECTED_JLD2 = "logged_packets_and_ytrain_phase_corrected.jld2"
const ANALYSIS_PILOT = 0.41

struct CandidateSpec
    name::String
    llr_scale::Float64
    pilot_llr::Float64
end

const CANDIDATES = CandidateSpec[
    CandidateSpec("bp_z_scale0.5_noclamp", 0.5, 0.0),
    CandidateSpec("bp_z_scale1_noclamp", 1.0, 0.0),
    CandidateSpec("bp_z_scale2_noclamp", 2.0, 0.0),
    CandidateSpec("bp_z_scale1_clamp20", 1.0, 20.0),
    CandidateSpec("bp_z_scale2_clamp20", 2.0, 20.0),
    CandidateSpec("bp_z_scale1_clamp50", 1.0, 50.0),
    CandidateSpec("bp_z_scale2_clamp50", 2.0, 50.0),
    CandidateSpec("bp_z_scale4_clamp50", 4.0, 50.0),
]

function resolve_row_range(num_rows::Int)
    if length(ARGS) >= 2
        first_row = parse(Int, ARGS[1])
        last_row = parse(Int, ARGS[2])
        return first_row:last_row
    end
    return 1:min(50, num_rows)
end

function sign_fix_with_pilots(z_opt::Vector{Float64}, pilot::Vector{Int}, x_pilot::Vector)
    score = sum(sign.(z_opt[pilot]) .* sign.(real.(x_pilot)))
    return score < 0 ? -z_opt : z_opt
end

function dfec_bp_bits(z_fixed::Vector{Float64}, H_sparse, parity_sets, col_sets,
                      pilot::Vector{Int}, x_pilot::Vector, spec::CandidateSpec)
    # The DFEC path uses symbol space (+1 / -1), while BP uses the
    # conventional LLR sign where positive favors bit 0. In this repo,
    # `modulate(1) = +1` and `modulate(0) = -1`, so we negate the symbol
    # evidence before handing it to BP.
    Lch = -spec.llr_scale .* z_fixed
    if spec.pilot_llr > 0
        Lch[pilot] .= -spec.pilot_llr .* real.(x_pilot)
    end
    dec = BPdecoder.prprp_decode(H_sparse, 0.5 .* Lch, 1.0, parity_sets, col_sets; max_iter=50)
    return dec.x_hat
end

function main()
    project_root = dirname(@__DIR__)
    data_dir = joinpath(project_root, DATA_DIR)
    jld2_path = joinpath(data_dir, PHASE_CORRECTED_JLD2)

    @load jld2_path all_ytrain_df all_packets_df

    row_range = resolve_row_range(nrow(all_packets_df))
    println("Analyzing packet rows $(first(row_range)):$(last(row_range)) at pilot fraction $(ANALYSIS_PILOT)")

    x_train = ExperimentCore.prepare_training_signals(NUM_TRAIN)
    x_datas = ExperimentCore.prepare_data_signals(
        D_NODES, T_NODES, NPC, NUM_TRAIN, NUM_DATA, GAP, nrow(all_ytrain_df), data_dir
    )
    y_train_matrix = ExperimentCore.extract_signal_matrix(all_ytrain_df)
    H_est_omp, H_est_mmse = ExperimentCore.estimate_channels(
        y_train_matrix, x_train, H_LEN, K_SPARSE, NOISE_VARIANCE
    )
    cache = ExperimentCore.prepare_cache(
        all_ytrain_df, all_packets_df, x_train, x_datas, H_est_omp, H_est_mmse
    )

    old_dir = pwd()
    code = cols = idrows = pilot = nothing
    try
        cd(data_dir)
        code, cols, idrows, pilot = LDPCJDPMemoized.initcode(
            D_NODES, T_NODES, NPC; pilot_row_fraction=ANALYSIS_PILOT
        )
    finally
        cd(old_dir)
    end

    H_sparse = LDPCJDPMemoized.get_H_sparse(code)
    n_code = code.n
    tr_len = length(x_train)
    pilot = sort(pilot)
    nonpilot = sort(setdiff(1:n_code, pilot))
    p_len = length(pilot)

    sums = Dict{String, Float64}("direct_dfec" => 0.0)
    for spec in CANDIDATES
        sums[spec.name] = 0.0
    end

    count = 0
    for packet_idx in row_range
        train_idx = cache.packet_to_train_idx[packet_idx]
        xdata_idx = cache.packet_to_xdata_idx[packet_idx]

        y_data = cache.packet_matrix[packet_idx, 1:n_code]

        h_full = cache.H_est_omp[train_idx, :]
        h_pos = findall(!iszero, h_full)
        h_init = h_full[h_pos]
        x_pilot = cache.x_datas[xdata_idx, pilot]
        pilot_data = LDPCJDPMemoized.demodulate.(x_pilot)
        pilot_bpsk = LDPCJDPMemoized.modulate.(pilot_data)

        _, _, result = LDPCJDPMemoized.decode_sparse_joint(
            y_data, code, idrows, pilot, pilot_bpsk, h_pos;
            h_init=h_init, λ=LAMBDA, γ=GAMMA, η=ETA, verbose=false
        )

        z_opt = Optim.minimizer(result)[1:n_code]
        z_fixed = sign_fix_with_pilots(z_opt, pilot, x_pilot)
        truth_symbols = sign.(real.(cache.x_datas[xdata_idx, nonpilot]))
        direct_symbols = sign.(z_fixed[nonpilot])
        sums["direct_dfec"] += sum(direct_symbols .!= truth_symbols) / n_code

        for spec in CANDIDATES
            bits_hat = dfec_bp_bits(z_fixed, H_sparse, idrows, cols, pilot, x_pilot, spec)
            sym_hat = sign.(real.(LDPCJDPMemoized.modulate.(bits_hat[nonpilot])))
            sums[spec.name] += sum(sym_hat .!= truth_symbols) / n_code
        end

        count += 1
        @printf("row %d done (%d/%d)\n", packet_idx, count, length(row_range))
    end

    println("\nMean BER over $(count) packet rows:")
    for (name, ber_sum) in sort(collect(sums); by=first)
        @printf("  %-22s %.6f\n", name, ber_sum / count)
    end
end

main()
