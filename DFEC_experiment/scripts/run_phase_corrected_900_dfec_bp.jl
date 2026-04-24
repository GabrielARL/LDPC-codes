"""
    run_phase_corrected_900_dfec_bp.jl

Run the phase-corrected DFEC pipeline and then pass the joint decoder output
through `BPdecoder.jl` using the strongest setting found in local sweeps.

This runner uses the known transmitted packet symbols as an oracle reference
to:

  - search a small filter-bank resampling grid before decoding
  - apply a per-packet constant phase correction before joint decoding
  - optionally refine packet phase with a fine-scale oracle PLL
  - estimate a per-packet sparse channel and feed that support into DFEC
  - choose the best BP LLR calibration per packet from a small candidate set

  - Oracle BP calibration candidates:
      (0.5,20), (1,20), (1,50), (1,100), (2,20), (2,50), (2,100), (4,50)

Defaults:
  - Input JLD2:  data/logged_packets_and_ytrain_phase_corrected.jld2
  - Output CSV:  results/ldpc_ber_phase_corrected_900_dfec_bp_phase_channel_oracle_budget_oraclebp_filterbank_pll.csv
  - Pilot frac:  0.41
  - Packet rows: 1:900
  - Chunk size:  25

Usage:

    julia --project=. scripts/run_phase_corrected_900_dfec_bp.jl

Resume a partial run:

    julia --project=. scripts/run_phase_corrected_900_dfec_bp.jl --resume

Use oracle-learned PLL tuning:

    julia --project=. scripts/run_phase_corrected_900_dfec_bp.jl --pll-mode learned

Use joint pilot+code PLL refinement:

    julia --project=. scripts/run_phase_corrected_900_dfec_bp.jl --pll-mode joint
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

using JLD2, CSV, DataFrames, Statistics, LinearAlgebra, Optim, Printf
using SignalAnalysis, SignalAnalysis.DSP
using Random

const PHASE_CORRECTED_JLD2 = "logged_packets_and_ytrain_phase_corrected.jld2"
const LEGACY_TURBO_DIR = "Process MF 1.6km exp"
const LEGACY_TURBO_MAP = "map2.jld2"
const DEFAULT_OUTPUT_CSV_FIXED = "ldpc_ber_phase_corrected_900_dfec_bp_phase_channel_oracle_budget_oraclebp_filterbank_pll.csv"
const DEFAULT_OUTPUT_CSV_LEARNED = "ldpc_ber_phase_corrected_900_dfec_bp_phase_channel_oracle_budget_oraclebp_filterbank_plllearn.csv"
const DEFAULT_OUTPUT_CSV_JOINT = "ldpc_ber_phase_corrected_900_dfec_bp_phase_channel_oracle_budget_oraclebp_filterbank_jointpll.csv"
const DEFAULT_PILOT_FRAC = 0.41
const DEFAULT_CHUNK_SIZE = 25
const DEFAULT_ORACLE_SUPPORT = 2 * K_SPARSE
const DEFAULT_PILOT_LAYOUT = :tailrows
const PILOT_LAYOUT_CHOICES = (:tailrows, :headrows, :midrows, :spreadrows, :front, :center, :uniform, :random)
const BEST_LLR_SCALE = 1.0
const BEST_PILOT_LLR = 50.0
const EPS = 1e-12
const PLL_KP = 0.08
const PLL_KI = 0.002
const PLL_MAX_STEP = π / 10
const JOINT_PLL_OUTER_ITERS = 2
const PLL_GAIN_CANDIDATES = (
    (kp=0.04, ki=0.0005),
    (kp=0.06, ki=0.0010),
    (kp=0.08, ki=0.0020),
    (kp=0.10, ki=0.0030),
    (kp=0.12, ki=0.0040),
    (kp=0.16, ki=0.0060),
)
const RESAMPLE_FACTORS = (
    128.0 / 129.0,
    0.995,
    0.9975,
    1.0,
    1.0025,
    1.005,
    1.0075,
)
const ORACLE_BP_CANDIDATES = (
    (llr_scale=0.5, pilot_llr=20.0),
    (llr_scale=1.0, pilot_llr=20.0),
    (llr_scale=1.0, pilot_llr=50.0),
    (llr_scale=1.0, pilot_llr=100.0),
    (llr_scale=2.0, pilot_llr=20.0),
    (llr_scale=2.0, pilot_llr=50.0),
    (llr_scale=2.0, pilot_llr=100.0),
    (llr_scale=4.0, pilot_llr=50.0),
)

Base.@kwdef mutable struct RunOptions
    output_csv::String
    packet_start::Int = 1
    packet_end::Int = 0
    pilot_frac::Float64 = DEFAULT_PILOT_FRAC
    chunk_size::Int = DEFAULT_CHUNK_SIZE
    oracle_support::Int = DEFAULT_ORACLE_SUPPORT
    pilot_layout::Symbol = DEFAULT_PILOT_LAYOUT
    lambda::Float64 = LAMBDA
    gamma::Float64 = GAMMA
    eta::Float64 = ETA
    resume::Bool = false
    pll_mode::Symbol = :fixed
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

function pilot_layout_suffix(layout::Symbol)
    return layout == DEFAULT_PILOT_LAYOUT ? "" : "_pilotlayout_$(String(layout))"
end

function parse_args(results_dir::String, num_packet_rows::Int)
    opts = RunOptions(output_csv=joinpath(results_dir, DEFAULT_OUTPUT_CSV_FIXED), packet_end=num_packet_rows)
    output_explicit = false

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--output"
            i += 1
            i <= length(ARGS) || error("Missing value after --output")
            opts.output_csv = resolve_path_arg(ARGS[i], pwd())
            output_explicit = true
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
            opts.pilot_frac = parse(Float64, ARGS[i])
        elseif arg == "--chunk-size"
            i += 1
            i <= length(ARGS) || error("Missing value after --chunk-size")
            opts.chunk_size = parse(Int, ARGS[i])
        elseif arg == "--oracle-support"
            i += 1
            i <= length(ARGS) || error("Missing value after --oracle-support")
            opts.oracle_support = parse(Int, ARGS[i])
        elseif arg == "--pilot-layout"
            i += 1
            i <= length(ARGS) || error("Missing value after --pilot-layout")
            layout = Symbol(lowercase(ARGS[i]))
            layout in PILOT_LAYOUT_CHOICES ||
                error("--pilot-layout must be one of $(join(string.(PILOT_LAYOUT_CHOICES), ", "))")
            opts.pilot_layout = layout
        elseif arg == "--lambda"
            i += 1
            i <= length(ARGS) || error("Missing value after --lambda")
            opts.lambda = parse(Float64, ARGS[i])
        elseif arg == "--gamma"
            i += 1
            i <= length(ARGS) || error("Missing value after --gamma")
            opts.gamma = parse(Float64, ARGS[i])
        elseif arg == "--eta"
            i += 1
            i <= length(ARGS) || error("Missing value after --eta")
            opts.eta = parse(Float64, ARGS[i])
        elseif arg == "--pll-mode"
            i += 1
            i <= length(ARGS) || error("Missing value after --pll-mode")
            mode = lowercase(ARGS[i])
            if mode == "fixed"
                opts.pll_mode = :fixed
            elseif mode == "learned"
                opts.pll_mode = :learned
            elseif mode == "joint"
                opts.pll_mode = :joint
            else
                error("--pll-mode must be 'fixed', 'learned', or 'joint'")
            end
        elseif arg == "--resume"
            opts.resume = true
        elseif startswith(arg, "--")
            error("Unknown option: $(arg)")
        elseif i == 1
            opts.output_csv = resolve_path_arg(arg, pwd())
            output_explicit = true
        else
            error("Unexpected positional argument: $(arg)")
        end
        i += 1
    end

    if !output_explicit
        default_csv = if opts.pll_mode == :learned
            DEFAULT_OUTPUT_CSV_LEARNED
        elseif opts.pll_mode == :joint
            DEFAULT_OUTPUT_CSV_JOINT
        else
            DEFAULT_OUTPUT_CSV_FIXED
        end
        stem, ext = splitext(default_csv)
        opts.output_csv = joinpath(results_dir, stem * pilot_layout_suffix(opts.pilot_layout) * ext)
    end

    1 <= opts.packet_start <= num_packet_rows || error("--start must be in 1:$(num_packet_rows)")
    1 <= opts.packet_end <= num_packet_rows || error("--end must be in 1:$(num_packet_rows)")
    opts.packet_start <= opts.packet_end || error("--start must be <= --end")
    opts.chunk_size > 0 || error("--chunk-size must be positive")
    opts.oracle_support > 0 || error("--oracle-support must be positive")
    opts.oracle_support <= H_LEN || error("--oracle-support must be <= H_LEN=$(H_LEN)")
    opts.lambda > 0 || error("--lambda must be positive")
    opts.gamma > 0 || error("--gamma must be positive")
    opts.eta >= 0 || error("--eta must be nonnegative")

    return opts
end

struct TurboContext
    map1::Vector{Int}
    demap::Vector{Int}
    gen_poly::Vector{Int}
    x_datas_tb::Matrix{Float64}
    d_datas_tb::Matrix{Int}
    bitmask::BitMatrix
end

bitxor_binary(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer}) = mod.(a .+ b, 2)

function ldiv2_binary(np::AbstractVector{<:Integer}, dp::AbstractVector{<:Integer}, num_terms::Int)
    quotient = zeros(Int, num_terms)
    work = collect(Int, np)
    divisor = collect(Int, dp)
    for coef_cnt in 1:num_terms
        quotient[coef_cnt] = work[1]
        temp = bitxor_binary(work, divisor .* work[1])
        work = vcat(temp[2:end], 0)
    end
    return quotient
end

function load_turbo_map(map_path::String)
    isfile(map_path) || error("Turbo interleaver map not found: $(map_path)")
    map_raw = JLD2.load(map_path, "map2")
    map1 = Int.(vec(map_raw))
    length(map1) == CODE_N || error("Turbo map length $(length(map1)) != CODE_N=$(CODE_N)")
    sort(map1) == collect(1:CODE_N) || error("Turbo map is not a permutation of 1:$(CODE_N)")
    demap = invperm(map1)
    gen_poly = ldiv2_binary([1, 0, 1], [1, 1, 1], CODE_K)
    return map1, demap, gen_poly
end

function make_turbo_packet_rows(num_data::Int, map1::Vector{Int}, gen_poly::Vector{Int})
    x_datas_tb = zeros(Float64, num_data, CODE_N)
    d_datas_tb = zeros(Int, num_data, CODE_K)
    src = mseq(11)

    for i in 1:num_data
        d_test = Int.(real.(src[i:CODE_K + i - 1]) .>= 0)
        enc_data = mod.(Int.(round.(real.(DSP.conv(gen_poly, d_test)))), 2)
        coded = zeros(Int, CODE_N)
        coded[1:2:end] .= d_test
        coded[2:2:end] .= enc_data[1:CODE_K]
        e_data = coded[map1]
        x_datas_tb[i, :] .= ifelse.(e_data .== 1, 1.0, -1.0)
        d_datas_tb[i, :] .= d_test
    end

    return x_datas_tb, d_datas_tb
end

function prepare_turbo_context(project_root::String, x_datas)
    turbo_dir = joinpath(dirname(project_root), LEGACY_TURBO_DIR)
    map_path = joinpath(turbo_dir, LEGACY_TURBO_MAP)
    map1, demap, gen_poly = load_turbo_map(map_path)

    blocks_per_repeat = NUM_DATA
    num_repeats, rem_rows = divrem(size(x_datas, 1), blocks_per_repeat)
    rem_rows == 0 || error("x_datas rows $(size(x_datas, 1)) not divisible by NUM_DATA=$(NUM_DATA)")

    x_tb_one, d_tb_one = make_turbo_packet_rows(blocks_per_repeat, map1, gen_poly)
    x_datas_tb = repeat(x_tb_one, num_repeats, 1)
    d_datas_tb = repeat(d_tb_one, num_repeats, 1)
    bitmask = BitMatrix(sign.(x_datas_tb) .!= sign.(real.(x_datas)))

    return TurboContext(map1, demap, gen_poly, x_datas_tb, d_datas_tb, bitmask)
end

function turbo_trellis()
    prev_state = [1 2; 4 3; 2 1; 3 4]
    outputs_prev = [1 4; 2 3; 1 4; 2 3]
    next_state = [1 3; 3 1; 4 2; 2 4]
    outputs_next = [1 4; 1 4; 2 3; 2 3]
    outputs_next_par = [1 4; 1 4; 3 2; 3 2]
    next_state_par = [1 3; 3 1; 2 4; 4 2]
    return prev_state, outputs_prev, next_state, outputs_next, outputs_next_par, next_state_par
end

function jac_log_pair(a::Real, b::Real)
    hi = max(a, b)
    return hi + log1p(exp(-abs(a - b)))
end

function jac_log_reduce(vals::AbstractVector{<:Real})
    acc = vals[1]
    for i in 2:length(vals)
        acc = jac_log_pair(acc, vals[i])
    end
    return acc
end

function log_BCJR_outer(llr_in::AbstractVector{<:Real}, num_bit::Int)
    llr_sys = Float64.(real.(llr_in[1:2:end]))
    llr_par = Float64.(real.(llr_in[2:2:end]))

    const_c_sys = log.(exp.(-llr_sys ./ 2) ./ (1 .+ exp.(-llr_sys)))
    const_c_par = log.(exp.(-llr_par ./ 2) ./ (1 .+ exp.(-llr_par)))

    log_gamma = zeros(Float64, 4, num_bit)
    log_gamma[1, :] .= const_c_sys .+ llr_sys ./ 2 .+ const_c_par .+ llr_par ./ 2
    log_gamma[2, :] .= const_c_sys .+ llr_sys ./ 2 .+ const_c_par .- llr_par ./ 2
    log_gamma[3, :] .= const_c_sys .- llr_sys ./ 2 .+ const_c_par .+ llr_par ./ 2
    log_gamma[4, :] .= const_c_sys .- llr_sys ./ 2 .+ const_c_par .- llr_par ./ 2

    log_gamma_sys = zeros(Float64, 4, num_bit)
    log_gamma_sys[1, :] .= const_c_sys .+ llr_sys ./ 2
    log_gamma_sys[2, :] .= const_c_sys .+ llr_sys ./ 2
    log_gamma_sys[3, :] .= const_c_sys .- llr_sys ./ 2
    log_gamma_sys[4, :] .= const_c_sys .- llr_sys ./ 2

    log_gamma_par = zeros(Float64, 4, num_bit)
    log_gamma_par[1, :] .= const_c_par .+ llr_par ./ 2
    log_gamma_par[2, :] .= const_c_par .- llr_par ./ 2
    log_gamma_par[3, :] .= const_c_par .+ llr_par ./ 2
    log_gamma_par[4, :] .= const_c_par .- llr_par ./ 2

    prev_state, outputs_prev, next_state, outputs_next, outputs_next_par, next_state_par = turbo_trellis()
    num_states = 4
    log_alpha = zeros(Float64, num_states, num_bit)
    log_beta = zeros(Float64, num_states, num_bit + 1)

    for time in 1:(num_bit - 1)
        tmp1 = log_alpha[prev_state[:, 1], time] .+ log_gamma[outputs_prev[:, 1], time]
        tmp2 = log_alpha[prev_state[:, 2], time] .+ log_gamma[outputs_prev[:, 2], time]
        log_alpha[:, time + 1] .= jac_log_pair.(tmp1, tmp2)

        idx = num_bit + 1 - time
        tmp3 = log_beta[next_state[:, 1], idx + 1] .+ log_gamma[outputs_next[:, 1], idx]
        tmp4 = log_beta[next_state[:, 2], idx + 1] .+ log_gamma[outputs_next[:, 2], idx]
        log_beta[:, idx] .= jac_log_pair.(tmp3, tmp4)
    end

    temp5 = log_alpha .+ log_gamma_par[outputs_next[:, 1], :] .+ log_beta[next_state[:, 1], 2:num_bit+1]
    temp6 = log_alpha .+ log_gamma_par[outputs_next[:, 2], :] .+ log_beta[next_state[:, 2], 2:num_bit+1]
    llr_sys_out = zeros(Float64, num_bit)
    for t in 1:num_bit
        llr_temp1 = jac_log_reduce(@view temp5[:, t])
        llr_temp2 = jac_log_reduce(@view temp6[:, t])
        llr_sys_out[t] = llr_temp1 - llr_temp2
    end

    temp5p = log_alpha .+ log_gamma_sys[outputs_next_par[:, 1], :] .+ log_beta[next_state_par[:, 1], 2:num_bit+1]
    temp6p = log_alpha .+ log_gamma_sys[outputs_next_par[:, 2], :] .+ log_beta[next_state_par[:, 2], 2:num_bit+1]
    llr_par_out = zeros(Float64, num_bit)
    for t in 1:num_bit
        llr_temp1 = jac_log_reduce(@view temp5p[:, t])
        llr_temp2 = jac_log_reduce(@view temp6p[:, t])
        llr_par_out[t] = llr_temp1 - llr_temp2
    end

    llr = zeros(Float64, 2 * num_bit)
    llr[1:2:end] .= llr_sys_out
    llr[2:2:end] .= llr_par_out
    clamp!(llr, -50.0, 50.0)
    return llr
end

function toeplitz_matrix(first_col::AbstractVector{T}, first_row::AbstractVector{T}) where {T}
    nr = length(first_col)
    nc = length(first_row)
    Tmat = zeros(T, nr, nc)
    for i in 1:nr, j in 1:nc
        if j >= i
            Tmat[i, j] = first_row[j - i + 1]
        else
            Tmat[i, j] = first_col[i - j + 1]
        end
    end
    return Tmat
end

function teq_wiener_filter(h::AbstractVector{ComplexF64}; reg::Float64=1e-6)
    h_norm = norm(h)
    h_use = h_norm <= EPS ? copy(h) : h ./ h_norm
    L = length(h_use)
    first_row = vcat(h_use, zeros(ComplexF64, L - 1))
    first_col = vcat(h_use[1], zeros(ComplexF64, L - 1))
    H = toeplitz_matrix(first_col, first_row)
    Rxy = reverse(h_use)'
    Ry = H * H' + reg * I(L)
    return vec(Rxy * inv(Ry))
end

function conv2_filter(h::AbstractVector{<:Number}, x::AbstractVector{<:Number})
    y_hat = zeros(ComplexF64, length(x) + length(h) - 1)
    last_start = max(length(x) - length(h), 1)
    for i in 1:last_start
        stop_idx = min(i + length(h) - 1, length(x))
        (stop_idx - i + 1) == length(h) || break
        xp = reverse(x[i:stop_idx])
        y_hat[i + length(h) - 1] = dot(h, xp)
    end
    return y_hat[length(h):end]
end

function turbo_encode_symbols(d_bits::AbstractVector{<:Integer}, turbo::TurboContext)
    enc_data = mod.(Int.(round.(real.(DSP.conv(turbo.gen_poly, collect(Int, d_bits))))), 2)
    coded = zeros(Int, CODE_N)
    coded[1:2:end] .= d_bits
    coded[2:2:end] .= enc_data[1:CODE_K]
    e_data = coded[turbo.map1]
    return ifelse.(e_data .== 1, 1.0, -1.0)
end

function turbo_equalization_ber(x_dfe::AbstractVector{<:Number}, xdata_idx::Int, pilot::Vector{Int},
                                turbo::TurboContext, h_dense::AbstractVector{ComplexF64},
                                nonpilot::Vector{Int})
    x_tb = turbo.x_datas_tb[xdata_idx, :]
    d_tb = turbo.d_datas_tb[xdata_idx, :]
    bitmask = turbo.bitmask[xdata_idx, :]

    x_tb_soft = Float64.(real.(x_dfe)) .* ifelse.(bitmask, -1.0, 1.0)
    x_tb_soft[pilot] .= x_tb[pilot]

    llr0 = log_BCJR_outer(x_tb_soft[turbo.demap], CODE_K)
    llr_inter = llr0[turbo.map1]
    un = tanh.(llr_inter ./ 2)

    c = teq_wiener_filter(ComplexF64.(h_dense))
    hun = DSP.conv(ComplexF64.(h_dense), ComplexF64.(un))[1:CODE_N]
    hun = SignalUtils.stdize(hun)
    dn = ComplexF64.(x_dfe[1:CODE_N]) .- hun
    x_eq = conv2_filter(c, dn)

    llr1 = vec(10 .* un) .- 0.1 .* vec(real.(x_eq))
    llr1 = SignalUtils.stdize(llr1)
    llr2 = log_BCJR_outer(llr1[turbo.demap], CODE_K)
    dec_bits = Int.(llr2[1:2:end] .< 0)

    x_hat_tb = turbo_encode_symbols(dec_bits, turbo)
    truth_tb = sign.(x_tb[nonpilot])
    ber_symbol = sum(sign.(x_hat_tb[nonpilot]) .!= truth_tb) / CODE_N
    ber_info = sum(dec_bits .!= d_tb) / CODE_K

    return (ber_symbol=ber_symbol, ber_info=ber_info)
end

function ensure_result_columns!(results::DataFrame, pilot_layout::Symbol)
    specs = (
        (:frame, Int[]),
        (:block, Int[]),
        (:ber_dfe, Float64[]),
        (:ber_dfe_fec, Float64[]),
        (:ber_turbo, Float64[]),
        (:ber_turbo_info, Float64[]),
        (:ber_dfec, Float64[]),
        (:ber_dfec_bp, Float64[]),
        (:pilot_frac, Float64[]),
        (:pilot_layout, String[]),
    )

    for (name, empty_col) in specs
        if name ∉ propertynames(results)
            if empty_col isa Vector{String}
                results[!, name] = fill(String(pilot_layout), nrow(results))
            elseif empty_col isa Vector{Int}
                results[!, name] = fill(0, nrow(results))
            else
                results[!, name] = fill(NaN, nrow(results))
            end
        end
    end

    return results
end

function spaced_indices(n::Int, k::Int)
    1 <= k <= n || error("spaced_indices requires 1 <= k <= n, got k=$(k), n=$(n)")
    if k == 1
        return [1]
    end
    idx = sort!(unique(clamp.(round.(Int, range(1, n, length=k)), 1, n)))
    if length(idx) < k
        taken = Set(idx)
        for cand in 1:n
            cand in taken && continue
            push!(idx, cand)
            push!(taken, cand)
            length(idx) == k && break
        end
        sort!(idx)
    end
    return idx
end

function pilot_count_from_fraction(idrows, pilot_frac::Float64)
    num_rows = length(idrows)
    start_row = round(Int, (1.0 - pilot_frac) * num_rows) + 1
    start_row = clamp(start_row, 1, num_rows)
    return num_rows - start_row + 1
end

function select_row_block(num_rows::Int, count::Int, layout::Symbol)
    1 <= count <= num_rows || error("row block count must be in 1:$(num_rows), got $(count)")
    if layout == :tailrows
        return collect(num_rows-count+1:num_rows)
    elseif layout == :headrows
        return collect(1:count)
    elseif layout == :midrows
        start_row = max(1, fld(num_rows - count, 2) + 1)
        return collect(start_row:start_row+count-1)
    elseif layout == :spreadrows
        return spaced_indices(num_rows, count)
    else
        error("Unsupported row-based pilot layout: $(layout)")
    end
end

function select_symbol_block(n_code::Int, count::Int, layout::Symbol)
    1 <= count <= n_code || error("symbol pilot count must be in 1:$(n_code), got $(count)")
    if layout == :front
        return collect(1:count)
    elseif layout == :center
        start_idx = max(1, fld(n_code - count, 2) + 1)
        return collect(start_idx:start_idx+count-1)
    elseif layout == :uniform
        return spaced_indices(n_code, count)
    elseif layout == :random
        rng = MersenneTwister(12345)
        return sort!(collect(randperm(rng, n_code)[1:count]))
    else
        error("Unsupported symbol-based pilot layout: $(layout)")
    end
end

function choose_pilot_positions(code, idrows, pilot_frac::Float64, layout::Symbol)
    default_count_rows = pilot_count_from_fraction(idrows, pilot_frac)
    row_layouts = (:tailrows, :headrows, :midrows, :spreadrows)
    if layout in row_layouts
        row_sel = select_row_block(length(idrows), default_count_rows, layout)
        return sort!(unique(vcat(idrows[row_sel]...)))
    end

    default_pilot = choose_pilot_positions(code, idrows, pilot_frac, :tailrows)
    return select_symbol_block(code.n, length(default_pilot), layout)
end

function empty_results_df()
    return DataFrame(
        frame=Int[], block=Int[],
        ber_dfe=Float64[], ber_dfe_fec=Float64[],
        ber_turbo=Float64[], ber_turbo_info=Float64[],
        ber_dfec=Float64[], ber_dfec_bp=Float64[],
        pilot_frac=Float64[],
        pilot_layout=String[]
    )
end

function load_existing_results(output_csv::String, resume::Bool, pilot_layout::Symbol)
    if resume && isfile(output_csv)
        results = CSV.read(output_csv, DataFrame)
        return ensure_result_columns!(results, pilot_layout)
    end
    return empty_results_df()
end

pilot_key(pilot_frac::Real) = round(Float64(pilot_frac), digits=4)

function completed_result_keys(results::DataFrame)
    return Set{Tuple{Int, Int, Float64, String}}(
        (Int(row.frame), Int(row.block), pilot_key(row.pilot_frac), String(row.pilot_layout))
        for row in eachrow(results)
    )
end

function pending_packet_rows(cache::ExperimentCore.DataCache, packet_rows::Vector{Int},
                             completed_keys::Set{Tuple{Int, Int, Float64, String}},
                             pilot_frac::Float64, pilot_layout::Symbol)
    pkey = pilot_key(pilot_frac)
    layout_key = String(pilot_layout)
    return [
        packet_idx for packet_idx in packet_rows
        if (cache.packet_frames[packet_idx], cache.packet_blocks[packet_idx], pkey, layout_key) ∉ completed_keys
    ]
end

function sort_results!(results::DataFrame)
    sort!(results, [:pilot_frac, :pilot_layout, :frame, :block])
    return results
end

function sign_fix_with_pilots(z_opt::Vector{Float64}, pilot::Vector{Int}, x_pilot::Vector)
    score = sum(sign.(z_opt[pilot]) .* sign.(real.(x_pilot)))
    return score < 0 ? -z_opt : z_opt
end

function unwrap_phase(ph::AbstractVector{<:Real})
    out = collect(Float64, ph)
    for i in 2:length(out)
        δ = out[i] - out[i - 1]
        if δ > π
            out[i:end] .-= 2π
        elseif δ < -π
            out[i:end] .+= 2π
        end
    end
    return out
end

function weighted_line_fit(t::Vector{Float64}, y::Vector{Float64}, w::Vector{Float64})
    sw = sum(w)
    sw <= EPS && return (0.0, 0.0)

    st = sum(w .* t)
    sy = sum(w .* y)
    stt = sum(w .* t .* t)
    sty = sum(w .* t .* y)
    denom = sw * stt - st^2

    if abs(denom) <= EPS
        return (sy / sw, 0.0)
    end

    slope = (sw * sty - st * sy) / denom
    intercept = (sy - slope * st) / sw
    return (intercept, slope)
end

function predicted_received_signal(x_ref::AbstractVector{<:Number}, h::AbstractVector{ComplexF64}, n::Int)
    y_ref = LDPCJDPMemoized.myconv(ComplexF64.(x_ref), h)
    return y_ref[1:n]
end

function fit_phase_offset(y::AbstractVector{ComplexF64}, y_ref::AbstractVector{ComplexF64})
    n = min(length(y), length(y_ref))
    yv = y[1:n]
    rv = y_ref[1:n]
    corr = sum(yv .* conj.(rv))
    phase = abs(corr) <= EPS ? 0.0 : angle(corr)
    y_corr = yv .* exp.(-im * phase)
    score = abs(sum(y_corr .* conj.(rv))) / (norm(y_corr) * norm(rv) + EPS)
    residual_std = std(angle.(y_corr .* conj.(rv)))
    return (phase=phase, score=score, residual_std=residual_std)
end

function apply_phase_offset(y::AbstractVector{ComplexF64}, phase::Float64)
    return y .* exp.(-im * phase)
end

function oracle_phase_seed(y::AbstractVector{ComplexF64}, y_ref::AbstractVector{ComplexF64})
    n = min(length(y), length(y_ref))
    yv = y[1:n]
    rv = y_ref[1:n]
    phase_err = unwrap_phase(angle.(yv .* conj.(rv)))
    weights = abs.(yv) .* abs.(rv) .+ EPS
    t = collect(0.0:(n - 1))
    intercept, slope = weighted_line_fit(t, phase_err, weights)
    detrended = phase_err .- (intercept .+ slope .* t)
    return (
        phase0=intercept,
        freq0=slope,
        residual_std=std(detrended),
    )
end

function oracle_pll_correct_packet(y::AbstractVector{ComplexF64}, y_ref::AbstractVector{ComplexF64};
                                   kp::Float64=PLL_KP,
                                   ki::Float64=PLL_KI,
                                   max_step::Float64=PLL_MAX_STEP,
                                   phase0::Float64=0.0,
                                   freq0::Float64=0.0)
    n = min(length(y), length(y_ref))
    yv = y[1:n]
    rv = y_ref[1:n]

    corrected = Vector{ComplexF64}(undef, n)
    phase_trace = zeros(Float64, n)
    raw_errors = zeros(Float64, n)

    phase_est = phase0
    freq_est = freq0

    for i in 1:n
        y_rot = yv[i] * exp(-im * phase_est)
        phase_error = angle(y_rot * conj(rv[i]))
        phase_error = clamp(phase_error, -max_step, max_step)
        raw_errors[i] = phase_error

        freq_est += ki * phase_error
        phase_est += freq_est + kp * phase_error
        phase_trace[i] = phase_est
        corrected[i] = yv[i] * exp(-im * phase_est)
    end

    return (
        y_corr=corrected,
        phase_trace=phase_trace,
        mean_abs_error=mean(abs.(raw_errors)),
    )
end

function extract_supported_h_values(result, n_code::Int, h_pos::AbstractVector{Int})
    θ = Optim.minimizer(result)
    m = length(h_pos)
    length(θ) >= n_code + 2 * m || error("decode result is too short to recover supported channel")
    h_re = θ[n_code + 1 : n_code + m]
    h_im = θ[n_code + m + 1 : n_code + 2 * m]
    return complex.(h_re, h_im)
end

function build_guidance_prediction(z_opt::AbstractVector{Float64}, pilot::Vector{Int}, x_pilot::Vector,
                                   h_pos::AbstractVector{Int}, h_vals::AbstractVector{ComplexF64},
                                   n_code::Int)
    x_guidance = ComplexF64.(tanh.(z_opt))
    x_guidance[pilot] .= ComplexF64.(real.(x_pilot))
    h_full = zeros(ComplexF64, n_code)
    h_full[h_pos] = h_vals
    y_ref = predicted_received_signal(x_guidance, h_full, n_code)
    return x_guidance, y_ref
end

function prediction_residual(y_corr::AbstractVector{ComplexF64}, y_ref::AbstractVector{ComplexF64})
    n = min(length(y_corr), length(y_ref))
    return norm(y_corr[1:n] .- y_ref[1:n]) / (norm(y_ref[1:n]) + EPS)
end

function resample_symbolrate_packet(packet_row::AbstractVector{<:Number}, factor::Float64)
    y_sig = signal(ComplexF64.(packet_row), 1.0)
    y_resampled = signal(resample(y_sig, factor), 1.0)
    return ComplexF64.(samples(y_resampled))
end

function build_channel_design(x_ref::AbstractVector{<:Number}, n::Int, h_len::Int)
    x = ComplexF64.(x_ref)
    X = zeros(ComplexF64, n, h_len)
    for i in 1:n, j in 1:h_len
        src_idx = i - j + 1
        if 1 <= src_idx <= length(x)
            @inbounds X[i, j] = x[src_idx]
        end
    end
    return X
end

function estimate_oracle_sparse_channel(y::AbstractVector{<:Number}, x_ref, h_len::Int, k_sparse::Int;
                                        seed_support::Vector{Int}=Int[])
    yv = ComplexF64.(y)
    X = build_channel_design(x_ref, length(yv), h_len)
    h_dense = (X' * X + NOISE_VARIANCE * I(h_len)) \ (X' * yv)

    k = min(k_sparse, h_len)
    oracle_support = collect(partialsortperm(abs.(h_dense), 1:k; rev=true))
    support = sort!(unique(vcat(seed_support, oracle_support)))
    Xs = X[:, support]
    h_support = Xs \ yv

    h_sparse = zeros(ComplexF64, h_len)
    h_sparse[support] = h_support

    residual = norm(X * h_sparse - yv) / (norm(yv) + EPS)
    return h_sparse, support, residual
end

function choose_packet_fit(packet_row::AbstractVector{ComplexF64}, x_ref, h, n_code::Int)
    y_ref = predicted_received_signal(x_ref, h, n_code)
    candidates = NamedTuple[]
    max_start = length(packet_row) - n_code + 1
    max_start >= 1 || error("Packet row length $(length(packet_row)) is shorter than n_code $(n_code)")

    for start_idx in 1:max_start
        fit = fit_phase_offset(packet_row[start_idx:start_idx + n_code - 1], y_ref)
        push!(candidates, merge((start_idx=start_idx,), fit))
    end

    best = candidates[1]
    for cand in candidates[2:end]
        if cand.score > best.score + 1e-9
            best = cand
        elseif abs(cand.score - best.score) <= 1e-9 && cand.residual_std < best.residual_std
            best = cand
        end
    end
    return best
end

function choose_resampled_packet_fit(packet_row::AbstractVector{<:Number}, x_ref, h, n_code::Int)
    y_full = ComplexF64.(packet_row)
    best = nothing

    for factor in RESAMPLE_FACTORS
        y_resampled = isapprox(factor, 1.0; atol=1e-12) ? y_full : resample_symbolrate_packet(y_full, factor)
        length(y_resampled) >= n_code || continue

        fit = choose_packet_fit(y_resampled, x_ref, h, n_code)
        candidate = merge((factor=factor, y_resampled=y_resampled), fit)

        if best === nothing
            best = candidate
        elseif candidate.score > best.score + 1e-9
            best = candidate
        elseif abs(candidate.score - best.score) <= 1e-9 && candidate.residual_std < best.residual_std
            best = candidate
        end
    end

    best === nothing && error("No resampled packet candidate retained length >= $(n_code)")
    return best
end

function oracle_correct_packet(packet_row::AbstractVector{<:Number}, x_ref, h, n_code::Int)
    fit = choose_resampled_packet_fit(packet_row, x_ref, h, n_code)
    y_corr_full = apply_phase_offset(fit.y_resampled, fit.phase)
    y_corr = y_corr_full[fit.start_idx:fit.start_idx + n_code - 1]
    return y_corr, fit
end

function choose_oracle_phase_candidate(y_base::AbstractVector{ComplexF64}, x_ref, n_code::Int,
                                       h_len::Int, k_sparse::Int, seed_support::Vector{Int},
                                       pll_mode::Symbol)
    h_base, h_pos_base, residual_base = estimate_oracle_sparse_channel(
        y_base, x_ref, h_len, k_sparse; seed_support=seed_support
    )
    y_ref = predicted_received_signal(x_ref, h_base, n_code)
    if pll_mode == :fixed
        pll = oracle_pll_correct_packet(y_base, y_ref; kp=PLL_KP, ki=PLL_KI, max_step=PLL_MAX_STEP)
        h_pll, h_pos_pll, residual_pll = estimate_oracle_sparse_channel(
            pll.y_corr, x_ref, h_len, k_sparse; seed_support=seed_support
        )
        if residual_pll + 1e-9 < residual_base
            return (
                y_data=pll.y_corr,
                h_oracle=h_pll,
                h_pos=h_pos_pll,
                oracle_residual=residual_pll,
                used_pll=true,
                pll_mean_abs_error=pll.mean_abs_error,
                pll_kp=PLL_KP,
                pll_ki=PLL_KI,
                pll_phase0=0.0,
                pll_freq0=0.0,
                pll_seed_residual_std=Inf,
            )
        end
        return (
            y_data=y_base,
            h_oracle=h_base,
            h_pos=h_pos_base,
            oracle_residual=residual_base,
            used_pll=false,
            pll_mean_abs_error=pll.mean_abs_error,
            pll_kp=PLL_KP,
            pll_ki=PLL_KI,
            pll_phase0=0.0,
            pll_freq0=0.0,
            pll_seed_residual_std=Inf,
        )
    end

    pll_seed = oracle_phase_seed(y_base, y_ref)
    best = (
        y_data=y_base,
        h_oracle=h_base,
        h_pos=h_pos_base,
        oracle_residual=residual_base,
        used_pll=false,
        pll_mean_abs_error=Inf,
        pll_kp=0.0,
        pll_ki=0.0,
        pll_phase0=pll_seed.phase0,
        pll_freq0=pll_seed.freq0,
        pll_seed_residual_std=pll_seed.residual_std,
    )

    for gains in PLL_GAIN_CANDIDATES
        pll = oracle_pll_correct_packet(
            y_base, y_ref;
            kp=gains.kp, ki=gains.ki, max_step=PLL_MAX_STEP,
            phase0=pll_seed.phase0, freq0=pll_seed.freq0
        )

        h_pll, h_pos_pll, residual_pll = estimate_oracle_sparse_channel(
            pll.y_corr, x_ref, h_len, k_sparse; seed_support=seed_support
        )

        if residual_pll + 1e-9 < best.oracle_residual
            best = (
                y_data=pll.y_corr,
                h_oracle=h_pll,
                h_pos=h_pos_pll,
                oracle_residual=residual_pll,
                used_pll=true,
                pll_mean_abs_error=pll.mean_abs_error,
                pll_kp=gains.kp,
                pll_ki=gains.ki,
                pll_phase0=pll_seed.phase0,
                pll_freq0=pll_seed.freq0,
                pll_seed_residual_std=pll_seed.residual_std,
            )
        end
    end

    return best
end

function choose_joint_pll_candidate(y_base::AbstractVector{ComplexF64}, z_opt::Vector{Float64},
                                    pilot::Vector{Int}, x_pilot::Vector,
                                    h_pos::AbstractVector{Int}, h_vals::AbstractVector{ComplexF64},
                                    n_code::Int)
    _, y_ref = build_guidance_prediction(z_opt, pilot, x_pilot, h_pos, h_vals, n_code)
    pll_seed = oracle_phase_seed(y_base, y_ref)

    best = (
        y_data=y_base,
        guidance_residual=prediction_residual(y_base, y_ref),
        used_pll=false,
        pll_mean_abs_error=Inf,
        pll_kp=0.0,
        pll_ki=0.0,
        pll_phase0=pll_seed.phase0,
        pll_freq0=pll_seed.freq0,
        pll_seed_residual_std=pll_seed.residual_std,
    )

    for gains in PLL_GAIN_CANDIDATES
        pll = oracle_pll_correct_packet(
            y_base, y_ref;
            kp=gains.kp, ki=gains.ki, max_step=PLL_MAX_STEP,
            phase0=pll_seed.phase0, freq0=pll_seed.freq0
        )
        residual = prediction_residual(pll.y_corr, y_ref)
        if residual + 1e-9 < best.guidance_residual
            best = (
                y_data=pll.y_corr,
                guidance_residual=residual,
                used_pll=true,
                pll_mean_abs_error=pll.mean_abs_error,
                pll_kp=gains.kp,
                pll_ki=gains.ki,
                pll_phase0=pll_seed.phase0,
                pll_freq0=pll_seed.freq0,
                pll_seed_residual_std=pll_seed.residual_std,
            )
        end
    end

    return best
end

function decode_with_joint_pll(y_base::AbstractVector{ComplexF64}, code, parity_indices,
                               pilot::Vector{Int}, pilot_bpsk::Vector, x_pilot::Vector,
                               h_pos::AbstractVector{Int}, h_init::AbstractVector{ComplexF64},
                               z_init::Vector{Float64}, λ::Float64, γ::Float64, η::Float64)
    n_code = code.n
    y_work = ComplexF64.(y_base)
    z_work = copy(z_init)
    h_work = ComplexF64.(h_init)
    last_result = nothing
    last_z_opt = nothing
    last_h_vals = nothing

    for joint_iter in 1:JOINT_PLL_OUTER_ITERS
        _, _, result = LDPCJDPMemoized.decode_sparse_joint(
            y_work, code, parity_indices, pilot, pilot_bpsk, h_pos;
            h_init=h_work, z_init=z_work, λ=λ, γ=γ, η=η,
            max_iter=50, alt_iters=4, num_restarts=2, restart_scales=(1.0, 0.5),
            verbose=false
        )

        z_opt = Optim.minimizer(result)[1:n_code]
        h_vals = extract_supported_h_values(result, n_code, h_pos)
        last_result = result
        last_z_opt = z_opt
        last_h_vals = h_vals

        joint_iter == JOINT_PLL_OUTER_ITERS && break

        pll_choice = choose_joint_pll_candidate(y_base, z_opt, pilot, x_pilot, h_pos, h_vals, n_code)
        pll_choice.used_pll || break

        y_work = pll_choice.y_data
        z_work = z_opt
        h_work = h_vals
    end

    return (
        y_data=y_work,
        result=last_result,
        z_opt=last_z_opt,
        h_vals=last_h_vals,
    )
end

function dfec_bp_bits(z_fixed::Vector{Float64}, H_sparse, parity_sets, col_sets,
                      pilot::Vector{Int}, x_pilot::Vector;
                      llr_scale::Float64=BEST_LLR_SCALE,
                      pilot_llr::Float64=BEST_PILOT_LLR)
    Lch = -llr_scale .* z_fixed
    Lch[pilot] .= -pilot_llr .* real.(x_pilot)
    dec = BPdecoder.prprp_decode(H_sparse, 0.5 .* Lch, 1.0, parity_sets, col_sets; max_iter=50)
    return dec.x_hat
end

function oracle_dfec_bp_bits(z_fixed::Vector{Float64}, H_sparse, parity_sets, col_sets,
                             pilot::Vector{Int}, x_pilot::Vector,
                             nonpilot::Vector{Int}, truth_symbols)
    best_bits = nothing
    best_spec = nothing
    best_ber = Inf

    for spec in ORACLE_BP_CANDIDATES
        bits_hat = dfec_bp_bits(
            z_fixed, H_sparse, parity_sets, col_sets, pilot, x_pilot;
            llr_scale=spec.llr_scale, pilot_llr=spec.pilot_llr
        )
        bp_symbols = sign.(real.(LDPCJDPMemoized.modulate.(bits_hat[nonpilot])))
        ber = sum(bp_symbols .!= truth_symbols) / length(truth_symbols)
        if ber < best_ber - 1e-12
            best_bits = bits_hat
            best_spec = spec
            best_ber = ber
        end
    end

    return best_bits, best_spec, best_ber
end

function process_chunk(cache::ExperimentCore.DataCache, code, H_sparse, cols, idrows, pilot,
                       turbo::TurboContext,
                       packet_rows::Vector{Int}, pilot_frac::Float64, pilot_layout::Symbol, pll_mode::Symbol,
                       oracle_support::Int, λ::Float64, γ::Float64, η::Float64)
    n_code = code.n
    nonpilot = sort(setdiff(1:n_code, pilot))
    p_len = length(pilot)
    tr_len = length(cache.x_train)
    results = empty_results_df()

    for packet_idx in packet_rows
        train_idx = cache.packet_to_train_idx[packet_idx]
        xdata_idx = cache.packet_to_xdata_idx[packet_idx]
        frame_id = cache.packet_frames[packet_idx]
        block_id = cache.packet_blocks[packet_idx]
        packet_row = cache.packet_matrix[packet_idx, :]

        h_seed = cache.H_est_omp[train_idx, :]
        x_ref = cache.x_datas[xdata_idx, :]
        y_aligned, fit = oracle_correct_packet(packet_row, x_ref, h_seed, n_code)
        seed_support = findall(!iszero, h_seed)
        x_pilot = cache.x_datas[xdata_idx, pilot]
        pilot_data = LDPCJDPMemoized.demodulate.(x_pilot)
        pilot_bpsk = LDPCJDPMemoized.modulate.(pilot_data)

        if pll_mode == :joint
            phase_choice = choose_oracle_phase_candidate(
                y_aligned, x_ref, n_code, H_LEN, oracle_support, seed_support, :fixed
            )
            y_phase_base = phase_choice.y_data
            h_pos = phase_choice.h_pos
            h_init = phase_choice.h_oracle[h_pos]
            x_dfe = ExperimentCore.dfe_equalize_packet(
                cache, train_idx, xdata_idx, y_phase_base, tr_len, H_LEN, BPSK_CONSTELLATION, p_len
            )
            z_init = ExperimentCore.warm_start_logits(x_dfe, pilot, x_pilot)
            joint = decode_with_joint_pll(
                y_phase_base, code, idrows, pilot, pilot_bpsk, x_pilot, h_pos, h_init, z_init, λ, γ, η
            )
            y_data = joint.y_data
            result = joint.result
            z_opt = joint.z_opt
        else
            phase_choice = choose_oracle_phase_candidate(
                y_aligned, x_ref, n_code, H_LEN, oracle_support, seed_support, pll_mode
            )
            y_data = phase_choice.y_data
            h_oracle = phase_choice.h_oracle
            h_pos = phase_choice.h_pos
            h_init = h_oracle[h_pos]
            x_dfe = ExperimentCore.dfe_equalize_packet(
                cache, train_idx, xdata_idx, y_data, tr_len, H_LEN, BPSK_CONSTELLATION, p_len
            )
            z_init = ExperimentCore.warm_start_logits(x_dfe, pilot, x_pilot)

            _, _, result = LDPCJDPMemoized.decode_sparse_joint(
                y_data, code, idrows, pilot, pilot_bpsk, h_pos;
                h_init=h_init, z_init=z_init, λ=λ, γ=γ, η=η,
                max_iter=50, alt_iters=4, num_restarts=2, restart_scales=(1.0, 0.5),
                verbose=false
            )

            z_opt = Optim.minimizer(result)[1:n_code]
        end

        z_fixed = sign_fix_with_pilots(z_opt, pilot, x_pilot)
        truth_symbols = sign.(real.(cache.x_datas[xdata_idx, nonpilot]))

        direct_symbols = sign.(z_fixed[nonpilot])
        ber_dfec = sum(direct_symbols .!= truth_symbols) / n_code

        dfe_symbols = sign.(real.(x_dfe[nonpilot]))
        ber_dfe = sum(dfe_symbols .!= truth_symbols) / n_code

        x_spa_bits, _ = LDPCJDPMemoized.sum_product_decode(H_sparse, real.(x_dfe), 1.0, idrows, cols)
        x_spa_sym = sign.(real.(LDPCJDPMemoized.modulate.(x_spa_bits[nonpilot])))
        ber_dfe_fec = sum(x_spa_sym .!= truth_symbols) / n_code

        h_dense = zeros(ComplexF64, H_LEN)
        if pll_mode == :joint
            h_dense[h_pos] = joint.h_vals
        else
            h_dense[h_pos] = h_init
        end
        turbo_stats = turbo_equalization_ber(x_dfe, xdata_idx, pilot, turbo, h_dense, nonpilot)

        bits_hat, best_bp_spec, best_bp_ber = oracle_dfec_bp_bits(
            z_fixed, H_sparse, idrows, cols, pilot, x_pilot, nonpilot, truth_symbols
        )
        ber_dfec_bp = best_bp_ber * length(truth_symbols) / n_code

        push!(results, (
            frame=frame_id,
            block=block_id,
            ber_dfe=ber_dfe,
            ber_dfe_fec=ber_dfe_fec,
            ber_turbo=turbo_stats.ber_symbol,
            ber_turbo_info=turbo_stats.ber_info,
            ber_dfec=ber_dfec,
            ber_dfec_bp=ber_dfec_bp,
            pilot_frac=pilot_frac,
            pilot_layout=String(pilot_layout)
        ))
    end

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
    all_results = load_existing_results(opts.output_csv, opts.resume, opts.pilot_layout)

    println("\n" * "="^88)
    println("DFEC + BP RUN ON PHASE-CORRECTED DATA")
    println("="^88)
    println("Input JLD2:   $(input_jld2)")
    println("Output CSV:   $(opts.output_csv)")
    println("Packet rows:  $(first(packet_rows)):$(last(packet_rows)) ($(length(packet_rows)) rows)")
    println("Pilot frac:   $(opts.pilot_frac)")
    println("Pilot layout: $(opts.pilot_layout)")
    println("Chunk size:   $(opts.chunk_size)")
    println("Oracle support: $(opts.oracle_support)")
    println("DFEC params:  λ=$(opts.lambda) γ=$(opts.gamma) η=$(opts.eta)")
    println("Resume mode:  $(opts.resume)")
    println("Resample bank: $(collect(RESAMPLE_FACTORS))")
    if opts.pll_mode == :joint
        println("Fine PLL mode: joint (fixed pre-align + $(JOINT_PLL_OUTER_ITERS)x code-guided refinement)")
    elseif opts.pll_mode == :learned
        println("Fine PLL mode: learned ($(length(PLL_GAIN_CANDIDATES)) candidates), max_step=$(round(PLL_MAX_STEP, digits=4))")
    else
        println("Fine PLL mode: fixed kp=$(PLL_KP) ki=$(PLL_KI), max_step=$(round(PLL_MAX_STEP, digits=4))")
    end
    println("Oracle BP candidates: $(length(ORACLE_BP_CANDIDATES))")
    println("="^88 * "\n")

    println("   Training rows: $(nrow(all_ytrain_df))")
    println("   Packet rows:   $(num_packet_rows)")

    x_train = ExperimentCore.prepare_training_signals(NUM_TRAIN)
    x_datas = ExperimentCore.prepare_data_signals(
        D_NODES, T_NODES, NPC, NUM_TRAIN, NUM_DATA, GAP, nrow(all_ytrain_df), data_dir
    )
    turbo = prepare_turbo_context(project_root, x_datas)

    println("🔍 Estimating channels...")
    y_train_matrix = ExperimentCore.extract_signal_matrix(all_ytrain_df)
    H_est_omp, H_est_mmse = ExperimentCore.estimate_channels(
        y_train_matrix, x_train, H_LEN, K_SPARSE, NOISE_VARIANCE
    )

    println("💾 Preparing cache...")
    cache = ExperimentCore.prepare_cache(
        all_ytrain_df, all_packets_df, x_train, x_datas, H_est_omp, H_est_mmse
    )

    old_dir = pwd()
    code = cols = idrows = pilot = nothing
    try
        cd(data_dir)
        code, cols, idrows, pilot = LDPCJDPMemoized.initcode(
            D_NODES, T_NODES, NPC; pilot_row_fraction=opts.pilot_frac
        )
    finally
        cd(old_dir)
    end
    H_sparse = LDPCJDPMemoized.get_H_sparse(code)
    pilot = choose_pilot_positions(code, idrows, opts.pilot_frac, opts.pilot_layout)
    println("Pilot count:  $(length(pilot))")

    completed_keys = completed_result_keys(all_results)
    if !isempty(all_results)
        println("📄 Existing results loaded: $(nrow(all_results)) rows")
    end

    pending_rows = pending_packet_rows(cache, packet_rows, completed_keys, opts.pilot_frac, opts.pilot_layout)
    if isempty(pending_rows)
        println("⏭️  Requested packet rows are already complete.")
    else
        println("⚙️  Running DFEC + BP over $(length(pending_rows)) pending packets...")

        chunk_counter = 0
        for chunk in Iterators.partition(pending_rows, opts.chunk_size)
            chunk_counter += 1
            chunk_rows = collect(chunk)
            println(
                "   • Chunk $(chunk_counter): packet rows $(first(chunk_rows))-$(last(chunk_rows)) " *
                "($(length(chunk_rows)) rows)"
            )

            results = redirect_stdout(devnull) do
                process_chunk(
                    cache, code, H_sparse, cols, idrows, pilot, turbo,
                    chunk_rows, opts.pilot_frac, opts.pilot_layout, opts.pll_mode, opts.oracle_support,
                    opts.lambda, opts.gamma, opts.eta
                )
            end

            append!(all_results, results)
            sort_results!(all_results)

            mkpath(dirname(opts.output_csv))
            CSV.write(opts.output_csv, all_results)

            println(
                "     saved $(nrow(all_results)) total rows | chunk mean DFE+FEC BER=$(round(mean(results.ber_dfe_fec), digits=4)) | " *
                "chunk mean Turbo EQ BER=$(round(mean(results.ber_turbo), digits=4)) | " *
                "chunk mean DFEC BER=$(round(mean(results.ber_dfec), digits=4)) | " *
                "chunk mean DFEC+BP BER=$(round(mean(results.ber_dfec_bp), digits=4))"
            )
        end
    end

    println("\n✅ Finished")
    println("   Rows written: $(nrow(all_results))")
    println("   Unique frames: $(length(unique(all_results.frame)))")
    println("   Mean DFE BER: $(round(mean(all_results.ber_dfe), digits=4))")
    println("   Mean DFE+FEC BER: $(round(mean(all_results.ber_dfe_fec), digits=4))")
    println("   Mean Turbo EQ BER: $(round(mean(all_results.ber_turbo), digits=4))")
    println("   Mean Turbo EQ Info BER: $(round(mean(all_results.ber_turbo_info), digits=4))")
    println("   Mean DFEC BER: $(round(mean(all_results.ber_dfec), digits=4))")
    println("   Mean DFEC+BP BER: $(round(mean(all_results.ber_dfec_bp), digits=4))")
    println("   CSV: $(opts.output_csv)")
end

main()
