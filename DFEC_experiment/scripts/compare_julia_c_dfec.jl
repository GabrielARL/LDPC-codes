#!/usr/bin/env julia

"""
    compare_julia_c_dfec.jl

Standalone comparison runner for the Julia DFEC implementation and the C
`joint_dfec_bp` executable on the exact same synthetic packet and received
samples.

Run from the repository root with:

    julia --project=DFEC_experiment DFEC_experiment/scripts/compare_julia_c_dfec.jl

Optional arguments:

    --seed 123
    --sigma 0.25
    --taps 1.0,0.35,-0.15
    --code 64,128,4
    --skip-build-c

The script also:
- compares Julia AD/manual gradients against a gradient dumped by the
  compiled C executable
- reports one-packet decode timing for Julia and C on the same received block
"""

using Printf
using Random
using SparseArrays
using LinearAlgebra
using ForwardDiff

module DFECCompareRuntime
include(joinpath(@__DIR__, "..", "src", "FEC.jl"))
include(joinpath(@__DIR__, "..", "src", "BPdecoder.jl"))
include(joinpath(@__DIR__, "..", "src", "LDPCJDPMemoized.jl"))
end

using .DFECCompareRuntime.LDPCJDPMemoized
using .DFECCompareRuntime.BPdecoder

const DEFAULT_SEED = 123
const DEFAULT_SIGMA = 0.25
const DEFAULT_TAPS = [1.0, 0.35, -0.15]
const DEFAULT_CODE = (64, 128, 4)
const DEFAULT_LAMBDA = 2.0
const DEFAULT_GAMMA = 1e-3
const DEFAULT_ETA = 1.0
const DEFAULT_ALT_ITERS = 4
const DEFAULT_Z_STEPS = 50
const DEFAULT_RESTARTS = 2
const DEFAULT_RESTART_SCALES = (1.0, 0.5)
const DEFAULT_BP_ITERS = 200
const DEFAULT_WARM_CLIP = 0.98

struct CompareOptions
    seed::Int
    sigma::Float64
    taps::Vector{Float64}
    code_k::Int
    code_n::Int
    code_npc::Int
    build_c::Bool
end

function parse_options(args::Vector{String})
    seed = DEFAULT_SEED
    sigma = DEFAULT_SIGMA
    taps = copy(DEFAULT_TAPS)
    code_k, code_n, code_npc = DEFAULT_CODE
    build_c = true

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--seed"
            i += 1
            i <= length(args) || error("Missing value after --seed")
            seed = parse(Int, args[i])
        elseif arg == "--sigma"
            i += 1
            i <= length(args) || error("Missing value after --sigma")
            sigma = parse(Float64, args[i])
        elseif arg == "--taps"
            i += 1
            i <= length(args) || error("Missing value after --taps")
            taps = parse.(Float64, split(args[i], ","))
        elseif arg == "--code"
            i += 1
            i <= length(args) || error("Missing value after --code")
            vals = parse.(Int, split(args[i], ","))
            length(vals) == 3 || error("--code must be k,n,npc")
            code_k, code_n, code_npc = vals
        elseif arg == "--skip-build-c"
            build_c = false
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end

    isempty(taps) && error("Need at least one tap")
    abs(taps[1]) > 1e-12 || error("First tap must be non-zero")
    code_n > code_k || error("Need n > k")

    return CompareOptions(seed, sigma, taps, code_k, code_n, code_npc, build_c)
end

function repo_root()
    return normpath(joinpath(@__DIR__, "..", ".."))
end

function data_dir()
    return normpath(joinpath(@__DIR__, "..", "data"))
end

function run_logged(cmd::Cmd; allow_failure::Bool=false)
    output_path = tempname()
    try
        open(output_path, "w") do io
            proc = run(pipeline(ignorestatus(cmd), stdout=io, stderr=io))
            code = proc.exitcode
            text = read(output_path, String)
            if code != 0 && !allow_failure
                error("Command failed ($(code)): $(join(cmd.exec, " "))\n$text")
            end
            return code, text
        end
    finally
        isfile(output_path) && rm(output_path; force=true)
    end
end

function load_code(k::Int, n::Int, npc::Int)
    old_dir = pwd()
    try
        cd(data_dir())
        return LDPCJDPMemoized.initcode(k, n, npc; pilot_row_fraction=0.0)
    finally
        cd(old_dir)
    end
end

function encode_with_data_dir(code, bits)
    old_dir = pwd()
    try
        cd(data_dir())
        return LDPCJDPMemoized.encode(code, bits)
    finally
        cd(old_dir)
    end
end

function dfe_warm_start(y::Vector{Float64}, taps::Vector{Float64}; clip::Float64=DEFAULT_WARM_CLIP)
    n = length(y)
    decisions = zeros(Float64, n)
    z0 = zeros(Float64, n)
    for i in 1:n
        fb = 0.0
        for j in 2:length(taps)
            idx = i - j + 1
            if idx >= 1
                fb += taps[j] * decisions[idx]
            end
        end
        eq = (y[i] - fb) / taps[1]
        decisions[i] = eq >= 0 ? 1.0 : -1.0
        z0[i] = atanh(clamp(eq, -clip, clip))
    end
    return z0
end

function transmit_isi_bpsk(bits::AbstractVector{<:Integer}, taps::Vector{Float64}, sigma::Float64, rng::AbstractRNG)
    n = length(bits)
    y = zeros(Float64, n)
    for i in 1:n
        sample = 0.0
        for j in 1:length(taps)
            idx = i - j + 1
            if idx >= 1
                sample += taps[j] * (bits[idx] == 1 ? 1.0 : -1.0)
            end
        end
        y[i] = sample + sigma * randn(rng)
    end
    return y
end

function write_received_file(path::String, y::Vector{Float64})
    open(path, "w") do io
        for (i, v) in enumerate(y)
            if i == 1
                @printf(io, "%+.12e", v)
            else
                @printf(io, " %+.12e", v)
            end
        end
        write(io, "\n")
    end
end

function write_double_file(path::String, values::AbstractVector{<:Real})
    open(path, "w") do io
        for (i, v) in enumerate(values)
            if i > 1
                write(io, ' ')
            end
            @printf(io, "%.12e", Float64(v))
        end
        write(io, "\n")
    end
end

function read_bit_file(path::String)
    bits = Int[]
    for line in eachline(path)
        for c in line
            if c == '0' || c == '1'
                push!(bits, c == '1' ? 1 : 0)
            end
        end
    end
    return bits
end

function read_double_file(path::String)
    values = Float64[]
    for line in eachline(path)
        for token in split(line)
            push!(values, parse(Float64, token))
        end
    end
    return values
end

function print_ui_summary(name::AbstractString, value)
    println("UI_SUMMARY ", name, "=", value)
end

function parse_c_block_summary(text::String)
    summary_re_new = r"(?m)^\s*\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+([0-9eE+\-.]+)\s+([0-9eE+\-.]+)\s+([0-9eE+\-.]+)\s+([0-9eE+\-.]+)\s+([0-9eE+\-.]+)\s+([0-9eE+\-.]+)\s+\[([^\]]*)\]\s*$"
    m = match(summary_re_new, text)
    if m !== nothing
        taps_text = strip(m.captures[7])
        taps = isempty(taps_text) ? Float64[] : parse.(Float64, split(taps_text, ","))
        return (
            objective = parse(Float64, m.captures[1]),
            data_loss = parse(Float64, m.captures[2]),
            parity_loss = parse(Float64, m.captures[3]),
            joint_sec = parse(Float64, m.captures[4]),
            bp_sec = parse(Float64, m.captures[5]),
            total_sec = parse(Float64, m.captures[6]),
            taps = taps,
        )
    end

    summary_re_old = r"(?m)^\s*\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+([0-9eE+\-.]+)\s+([0-9eE+\-.]+)\s+([0-9eE+\-.]+)\s+\[([^\]]*)\]\s*$"
    m = match(summary_re_old, text)
    m === nothing && return nothing

    taps_text = strip(m.captures[4])
    taps = isempty(taps_text) ? Float64[] : parse.(Float64, split(taps_text, ","))
    return (
        objective = parse(Float64, m.captures[1]),
        data_loss = parse(Float64, m.captures[2]),
        parity_loss = parse(Float64, m.captures[3]),
        joint_sec = NaN,
        bp_sec = NaN,
        total_sec = NaN,
        taps = taps,
    )
end

function syndrome_weight(bits::Vector{Int}, H::SparseMatrixCSC{Bool,Int})
    syn = H * bits
    return count(!iszero, syn .% 2)
end

function compare_bits(a::Vector{Int}, b::Vector{Int})
    length(a) == length(b) || error("Bit lengths differ: $(length(a)) vs $(length(b))")
    return count(i -> a[i] != b[i], eachindex(a))
end

function count_oddr_rows(code)
    return count(sum(row) % 2 == 1 for row in eachrow(code.H))
end

function conv_supported(x::AbstractVector, h::AbstractVector)
    n = length(x)
    L = length(h)
    T = promote_type(eltype(x), eltype(h))
    yhat = zeros(T, n)
    for i in 1:n
        s = zero(T)
        for j in 1:min(i, L)
            s += h[j] * x[i - j + 1]
        end
        yhat[i] = s
    end
    return yhat
end

function julia_style_parity_loss(x::AbstractVector, parity_indices)
    T = eltype(x)
    loss = zero(T)
    for inds in parity_indices
        p = one(T)
        for idx in inds
            p *= x[idx]
        end
        diff = one(T) - p
        loss += diff * diff
    end
    return loss
end

function c_row_targets(parity_indices)
    return [isodd(length(inds)) ? -1.0 : 1.0 for inds in parity_indices]
end

function c_style_parity_loss(x::AbstractVector, parity_indices, row_targets::AbstractVector)
    T = promote_type(eltype(x), eltype(row_targets))
    loss = zero(T)
    for (inds, target) in zip(parity_indices, row_targets)
        p = one(T)
        for idx in inds
            p *= x[idx]
        end
        diff = p - T(target)
        loss += diff * diff
    end
    return loss
end

function julia_style_objective_ad(z::AbstractVector, y::AbstractVector, h::AbstractVector,
                                  h_prior::AbstractVector, parity_indices,
                                  λ::Real, γ::Real, η::Real)
    x = tanh.(z)
    res = conv_supported(x, h) .- y
    return sum(abs2, res) +
           λ * julia_style_parity_loss(x, parity_indices) +
           γ * (sum(abs2, z) + sum(abs2, h)) +
           η * sum(abs2, h .- h_prior)
end

function c_style_objective_ad(z::AbstractVector, y::AbstractVector, h::AbstractVector,
                              h_prior::AbstractVector, parity_indices,
                              row_targets::AbstractVector,
                              λ::Real, γ::Real, η::Real)
    x = tanh.(z)
    res = conv_supported(x, h) .- y
    return sum(abs2, res) +
           λ * c_style_parity_loss(x, parity_indices, row_targets) +
           γ * (sum(abs2, z) + sum(abs2, h)) +
           η * sum(abs2, h .- h_prior)
end

function c_style_manual_gradient(z::Vector{Float64}, y::Vector{Float64}, h::Vector{Float64},
                                 parity_indices, row_targets::Vector{Float64},
                                 λ::Float64, γ::Float64)
    n = length(z)
    x = tanh.(z)
    res = conv_supported(x, h) .- y
    grad = zeros(Float64, n)

    for tap in eachindex(h)
        for i in 1:(n - tap + 1)
            grad[i] += 2.0 * res[i + tap - 1] * h[tap]
        end
    end

    for i in eachindex(grad)
        grad[i] = grad[i] * (1.0 - x[i]^2) + 2.0 * γ * z[i]
    end

    for (inds, target) in zip(parity_indices, row_targets)
        d = length(inds)
        d == 0 && continue

        x_inds = x[inds]
        p = prod(x_inds)
        prefix = ones(Float64, d)
        suffix = ones(Float64, d)
        for j in 2:d
            prefix[j] = prefix[j - 1] * x_inds[j - 1]
        end
        for j in (d - 1):-1:1
            suffix[j] = suffix[j + 1] * x_inds[j + 1]
        end

        diff = p - target
        for (local_idx, global_idx) in enumerate(inds)
            prod_others = prefix[local_idx] * suffix[local_idx]
            grad[global_idx] += λ * 2.0 * diff * prod_others * (1.0 - x[global_idx]^2)
        end
    end

    return grad
end

function julia_style_manual_gradient(z::Vector{Float64}, y::Vector{Float64}, h::Vector{Float64},
                                     parity_indices, λ::Float64, γ::Float64)
    n = length(z)
    x = tanh.(z)
    res = conv_supported(x, h) .- y
    grad = zeros(Float64, n)

    for tap in eachindex(h)
        for i in 1:(n - tap + 1)
            grad[i] += 2.0 * res[i + tap - 1] * h[tap]
        end
    end

    for i in eachindex(grad)
        grad[i] = grad[i] * (1.0 - x[i]^2) + 2.0 * γ * z[i]
    end

    for inds in parity_indices
        d = length(inds)
        d == 0 && continue

        x_inds = x[inds]
        p = prod(x_inds)
        prefix = ones(Float64, d)
        suffix = ones(Float64, d)
        for j in 2:d
            prefix[j] = prefix[j - 1] * x_inds[j - 1]
        end
        for j in (d - 1):-1:1
            suffix[j] = suffix[j + 1] * x_inds[j + 1]
        end

        diff = 1.0 - p
        for (local_idx, global_idx) in enumerate(inds)
            prod_others = prefix[local_idx] * suffix[local_idx]
            grad[global_idx] += λ * 2.0 * diff * (-prod_others) * (1.0 - x[global_idx]^2)
        end
    end

    return grad
end

function run_julia_joint_bp(y::Vector{Float64}, code, H_sparse, cols, idrows,
                            taps::Vector{Float64}, z0::Vector{Float64})
    joint_t0 = time_ns()
    _, _, joint_result = LDPCJDPMemoized.decode_sparse_joint(
        y,
        code,
        idrows,
        Int[],
        ComplexF64[],
        collect(1:length(taps));
        λ = DEFAULT_LAMBDA,
        γ = DEFAULT_GAMMA,
        η = DEFAULT_ETA,
        h_init = ComplexF64.(taps),
        z_init = z0,
        max_iter = DEFAULT_Z_STEPS,
        alt_iters = DEFAULT_ALT_ITERS,
        num_restarts = DEFAULT_RESTARTS,
        restart_scales = DEFAULT_RESTART_SCALES,
        verbose = false,
    )
    joint_ns = time_ns() - joint_t0

    θ = getfield(joint_result, :θ)
    n = code.n
    L = length(taps)
    h_raw = θ[n + 1:n + L] .+ im .* θ[n + L + 1:n + 2L]

    bp_t0 = time_ns()
    bp = BPdecoder.prprp_decode(
        H_sparse,
        Float64.(-θ[1:n]),
        1.0,
        idrows,
        cols;
        max_iter = DEFAULT_BP_ITERS,
        stop_on_valid = true,
    )
    bp_ns = time_ns() - bp_t0

    return (
        result = joint_result,
        θ = θ,
        h_raw = h_raw,
        bp = bp,
        bits = Int.(bp.x_hat),
        joint_ns = joint_ns,
        bp_ns = bp_ns,
        total_ns = joint_ns + bp_ns,
    )
end

function main()
    opts = parse_options(ARGS)
    root = repo_root()

    println("=== Julia vs C DFEC Comparison ===")
    println("Repo root:  $root")
    println("Seed:       $(opts.seed)")
    println("Sigma:      $(opts.sigma)")
    println("Taps:       $(join(opts.taps, ", "))")
    println("Code:       $(opts.code_k)-$(opts.code_n)-$(opts.code_npc)")

    if opts.build_c
        println("\nBuilding C executables with make progs ...")
        run_logged(Cmd(Cmd(["make", "progs"]); dir=root))
    end

    c_exe = joinpath(root, "joint_dfec_bp")
    isfile(c_exe) || error("C executable not found at $c_exe")

    code, cols, idrows, _ = load_code(opts.code_k, opts.code_n, opts.code_npc)
    H_sparse = LDPCJDPMemoized.get_H_sparse(code)
    odd_rows = count_oddr_rows(code)
    if odd_rows > 0
        println("\nWarning: code has $odd_rows odd-weight parity rows.")
        println("Julia's parity penalty assumes target product +1, while the C code handles row-degree parity explicitly.")
        println("You may therefore see objective or decode differences that come from the model mismatch, not just implementation bugs.")
    end

    rng = MersenneTwister(opts.seed)
    msg_bits = rand(rng, 0:1, code.k)
    codeword = Int.(encode_with_data_dir(code, msg_bits))
    y = transmit_isi_bpsk(codeword, opts.taps, opts.sigma, rng)

    z0 = dfe_warm_start(y, opts.taps)
    println("\nWarming up Julia joint-DFEC + BP path to exclude first-call JIT from timing ...")
    run_julia_joint_bp(y, code, H_sparse, cols, idrows, opts.taps, z0)

    julia_run = run_julia_joint_bp(y, code, H_sparse, cols, idrows, opts.taps, z0)
    julia_result = julia_run.result
    θ = julia_run.θ
    n = code.n
    L = length(opts.taps)
    julia_h_raw = julia_run.h_raw
    julia_bp = julia_run.bp
    julia_bits_bp = julia_run.bits

    temp_dir = mktempdir()
    rec_path = joinpath(temp_dir, "packet.rec")
    dec_path = joinpath(temp_dir, "packet.dec")
    probe_dec_path = joinpath(temp_dir, "packet_probe.dec")
    z_path = joinpath(temp_dir, "julia_z.txt")
    h_path = joinpath(temp_dir, "julia_h.txt")
    grad_path = joinpath(temp_dir, "c_grad.txt")
    pchk_path = joinpath(data_dir(), "$(opts.code_k)-$(opts.code_n)-$(opts.code_npc).pchk")
    isfile(pchk_path) || error("Missing pchk file: $pchk_path")
    write_received_file(rec_path, y)

    c_cmd = Cmd(Cmd(vcat([c_exe, "-t", pchk_path, rec_path, dec_path], string.(opts.taps))); dir=root)
    c_wall_t0 = time_ns()
    c_exit, c_text = run_logged(c_cmd; allow_failure=true)
    c_wall_ns = time_ns() - c_wall_t0
    c_exit == 0 || error("C decoder failed:\n$c_text")
    c_bits = read_bit_file(dec_path)
    c_summary = parse_c_block_summary(c_text)

    z_eval = Float64.(θ[1:n])
    h_eval = Float64.(real.(julia_h_raw))
    write_double_file(z_path, z_eval)
    write_double_file(h_path, h_eval)

    c_grad_cmd = Cmd(Cmd(vcat([
        c_exe, "-t", "-b", "1", "-o", "1", "-j", "1", "-r", "1",
        "-G", grad_path, "-Z", z_path, "-H", h_path,
        pchk_path, rec_path, probe_dec_path
    ], string.(opts.taps))); dir=root)
    c_grad_exit, c_grad_text = run_logged(c_grad_cmd; allow_failure=true)
    c_grad_exit == 0 || error("C gradient probe failed:\n$c_grad_text")
    c_grad_pure = read_double_file(grad_path)
    length(c_grad_pure) == n || error("C gradient length mismatch: $(length(c_grad_pure)) vs $n")

    println("\n--- Results ---")
    println("Truth codeword valid:   ", LDPCJDPMemoized.is_valid_codeword(codeword, H_sparse))
    println("Julia BP valid:         ", BPdecoder.is_valid_codeword(julia_bits_bp, H_sparse))
    println("C decode valid:         ", BPdecoder.is_valid_codeword(c_bits, H_sparse))
    println("Julia BP iters:         ", julia_bp.iters)
    println("Julia joint objective:  ", @sprintf("%.6e", getfield(julia_result, :objective)))
    if c_summary !== nothing
        println("C joint objective:      ", @sprintf("%.6e", c_summary.objective))
    else
        println("C joint objective:      <not parsed>")
    end

    julia_err_truth = compare_bits(julia_bits_bp, codeword)
    c_err_truth = compare_bits(c_bits, codeword)
    julia_vs_c = compare_bits(julia_bits_bp, c_bits)

    println("\n--- Bit Differences ---")
    println("Julia BP vs truth:      $julia_err_truth")
    println("C decode vs truth:      $c_err_truth")
    println("Julia BP vs C decode:   $julia_vs_c")
    println("Julia syndrome weight:  ", syndrome_weight(julia_bits_bp, H_sparse))
    println("C syndrome weight:      ", syndrome_weight(c_bits, H_sparse))

    println("\n--- Timing (One Packet) ---")
    println("Julia timing note:      excludes first-call JIT via one warm-up run on the same packet")
    println("Julia joint DFEC:       ", @sprintf("%.3f ms", julia_run.joint_ns / 1e6))
    println("Julia BP:               ", @sprintf("%.3f ms", julia_run.bp_ns / 1e6))
    println("Julia total:            ", @sprintf("%.3f ms", julia_run.total_ns / 1e6))
    if c_summary !== nothing && !isnan(c_summary.total_sec)
        println("C joint DFEC:           ", @sprintf("%.3f ms", 1e3 * c_summary.joint_sec))
        println("C BP:                   ", @sprintf("%.3f ms", 1e3 * c_summary.bp_sec))
        println("C total (internal):     ", @sprintf("%.3f ms", 1e3 * c_summary.total_sec))
        println("Julia/C total ratio:    ", @sprintf("%.3fx", (julia_run.total_ns * 1e-9) / c_summary.total_sec))
    else
        println("C total (internal):     <not parsed>")
    end
    println("C command wall time:    ", @sprintf("%.3f ms", c_wall_ns / 1e6))

    println("\n--- Tap Comparison ---")
    println("Tap prior:              [", join(map(x -> @sprintf("%.6f", x), opts.taps), ", "), "]")
    println("Julia raw taps:         [", join(map(x -> @sprintf("%.6f%+.6fim", real(x), imag(x)), julia_h_raw), ", "), "]")
    if c_summary !== nothing
        println("C raw taps:             [", join(map(x -> @sprintf("%.6f", x), c_summary.taps), ", "), "]")
        if length(c_summary.taps) == length(julia_h_raw)
            tap_diff = norm(real.(julia_h_raw) .- c_summary.taps)
            println("Tap L2 difference:      ", @sprintf("%.6e", tap_diff))
        end
    end

    h_prior_real = Float64.(opts.taps)
    row_targets = c_row_targets(idrows)

    grad_julia_ad = ForwardDiff.gradient(z -> julia_style_objective_ad(
        z, y, h_eval, h_prior_real, idrows, DEFAULT_LAMBDA, DEFAULT_GAMMA, DEFAULT_ETA
    ), z_eval)
    grad_julia_manual = julia_style_manual_gradient(
        z_eval, y, h_eval, idrows, DEFAULT_LAMBDA, DEFAULT_GAMMA
    )
    grad_c_ad = ForwardDiff.gradient(z -> c_style_objective_ad(
        z, y, h_eval, h_prior_real, idrows, row_targets, DEFAULT_LAMBDA, DEFAULT_GAMMA, DEFAULT_ETA
    ), z_eval)
    grad_c_manual = c_style_manual_gradient(
        z_eval, y, h_eval, idrows, row_targets, DEFAULT_LAMBDA, DEFAULT_GAMMA
    )

    diff_julia_ad_vs_manual = grad_julia_ad .- grad_julia_manual
    diff_julia_vs_cad = grad_julia_ad .- grad_c_ad
    diff_cad_vs_manual = grad_c_ad .- grad_c_manual
    diff_cpure_vs_cmanual = c_grad_pure .- grad_c_manual
    diff_cpure_vs_cad = c_grad_pure .- grad_c_ad
    diff_julia_vs_cpure = grad_julia_ad .- c_grad_pure

    println("\n--- Gradient Comparison (AD) ---")
    println("Evaluation state:       Julia z_opt with real(Julia tap estimate)")
    println("||∇Julia_AD - ∇Julia_manual||₂: ", @sprintf("%.6e", norm(diff_julia_ad_vs_manual)))
    println("max|Julia_AD-Julia_manual|:     ", @sprintf("%.6e", maximum(abs.(diff_julia_ad_vs_manual))))
    println("||∇Julia_AD - ∇C_AD||₂: ", @sprintf("%.6e", norm(diff_julia_vs_cad)))
    println("max|Julia_AD-C_AD|:     ", @sprintf("%.6e", maximum(abs.(diff_julia_vs_cad))))
    println("||∇C_AD - ∇C_manual||₂: ", @sprintf("%.6e", norm(diff_cad_vs_manual)))
    println("max|C_AD-C_manual|:     ", @sprintf("%.6e", maximum(abs.(diff_cad_vs_manual))))
    println("||∇C_pure - ∇C_manual||₂: ", @sprintf("%.6e", norm(diff_cpure_vs_cmanual)))
    println("max|C_pure-C_manual|:     ", @sprintf("%.6e", maximum(abs.(diff_cpure_vs_cmanual))))
    println("||∇C_pure - ∇C_AD||₂: ", @sprintf("%.6e", norm(diff_cpure_vs_cad)))
    println("max|C_pure-C_AD|:     ", @sprintf("%.6e", maximum(abs.(diff_cpure_vs_cad))))
    println("||∇Julia_AD - ∇C_pure||₂: ", @sprintf("%.6e", norm(diff_julia_vs_cpure)))
    println("max|Julia_AD-C_pure|:     ", @sprintf("%.6e", maximum(abs.(diff_julia_vs_cpure))))
    if odd_rows > 0
        println("Note: gradient mismatch between Julia_AD and C_* includes the odd-row parity-target difference.")
    end

    print_ui_summary("julia_joint_ms", @sprintf("%.6f", julia_run.joint_ns / 1e6))
    print_ui_summary("julia_bp_ms", @sprintf("%.6f", julia_run.bp_ns / 1e6))
    print_ui_summary("julia_total_ms", @sprintf("%.6f", julia_run.total_ns / 1e6))
    if c_summary !== nothing && !isnan(c_summary.joint_sec)
        print_ui_summary("c_joint_ms", @sprintf("%.6f", 1e3 * c_summary.joint_sec))
        print_ui_summary("c_bp_ms", @sprintf("%.6f", 1e3 * c_summary.bp_sec))
        print_ui_summary("c_total_ms", @sprintf("%.6f", 1e3 * c_summary.total_sec))
        print_ui_summary("julia_c_ratio", @sprintf("%.6f", (julia_run.total_ns * 1e-9) / c_summary.total_sec))
    end
    print_ui_summary("c_wall_ms", @sprintf("%.6f", c_wall_ns / 1e6))
    print_ui_summary("julia_valid", BPdecoder.is_valid_codeword(julia_bits_bp, H_sparse) ? "1" : "0")
    print_ui_summary("c_valid", BPdecoder.is_valid_codeword(c_bits, H_sparse) ? "1" : "0")
    print_ui_summary("julia_vs_truth", julia_err_truth)
    print_ui_summary("c_vs_truth", c_err_truth)
    print_ui_summary("julia_vs_c", julia_vs_c)
    print_ui_summary("grad_cpure_vs_cmanual_l2", @sprintf("%.6e", norm(diff_cpure_vs_cmanual)))
    print_ui_summary("grad_cpure_vs_cad_l2", @sprintf("%.6e", norm(diff_cpure_vs_cad)))
    print_ui_summary("grad_julia_vs_cpure_l2", @sprintf("%.6e", norm(diff_julia_vs_cpure)))
    print_ui_summary("odd_row_count", odd_rows)

    if julia_vs_c == 0
        println("\nNo final decoded-bit difference found on this packet.")
    else
        diff_positions = findall(i -> julia_bits_bp[i] != c_bits[i], eachindex(c_bits))
        preview = first(diff_positions, min(length(diff_positions), 16))
        println("\nDecoded bits differ at $(length(diff_positions)) positions.")
        println("First differing positions: ", join(preview, ", "))
    end

    println("\nTemporary files kept in: $temp_dir")
    println("C decoder log:")
    println(c_text)
end

main()
