"""
    sweep_first_chunk_dfec_hyperparams.jl

Sweep DFEC objective hyperparameters over the first chunk (packet rows 1:25)
with oracle channel support fixed to a chosen value.

Defaults:
  - Support: 8
  - Pilot frac: 0.41
  - PLL mode: joint
  - Packet rows: 1:25
  - Lambda grid: 1.0, 2.0, 4.0
  - Gamma grid: 1e-4, 1e-3, 1e-2
  - Eta grid: 0.3, 1.0, 3.0

Usage:

    julia --project=. scripts/sweep_first_chunk_dfec_hyperparams.jl

Resume partial runs:

    julia --project=. scripts/sweep_first_chunk_dfec_hyperparams.jl --resume

Set explicit grids:

    julia --project=. scripts/sweep_first_chunk_dfec_hyperparams.jl \
      --lambdas 1,2,4 --gammas 1e-4,1e-3,1e-2 --etas 0.3,1,3
"""

using Printf
using CSV, DataFrames, Statistics

Base.@kwdef mutable struct SweepOptions
    lambdas::Vector{Float64} = [1.0, 2.0, 4.0]
    gammas::Vector{Float64} = [1e-4, 1e-3, 1e-2]
    etas::Vector{Float64} = [0.3, 1.0, 3.0]
    pilot_frac::Float64 = 0.41
    pll_mode::String = "joint"
    oracle_support::Int = 8
    packet_start::Int = 1
    packet_end::Int = 25
    chunk_size::Int = 25
    resume::Bool = false
    output_dir::String = ""
    summary_csv::String = ""
end

function parse_float_list(spec::String)
    vals = Float64[]
    for item in split(spec, ',')
        stripped = strip(item)
        isempty(stripped) && continue
        push!(vals, parse(Float64, stripped))
    end
    isempty(vals) && error("No values parsed")
    return vals
end

function fmt_float(x::Float64)
    s = @sprintf("%.6g", x)
    return replace(s, "." => "p", "-" => "m", "+" => "")
end

function parse_args(project_root::String)
    outdir = joinpath(project_root, "results", "dfec_hparam_sweep_chunk1")
    opts = SweepOptions(output_dir=outdir, summary_csv=joinpath(outdir, "dfec_hparam_sweep_summary.csv"))

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--lambdas"
            i += 1
            i <= length(ARGS) || error("Missing value after --lambdas")
            opts.lambdas = parse_float_list(ARGS[i])
        elseif arg == "--gammas"
            i += 1
            i <= length(ARGS) || error("Missing value after --gammas")
            opts.gammas = parse_float_list(ARGS[i])
        elseif arg == "--etas"
            i += 1
            i <= length(ARGS) || error("Missing value after --etas")
            opts.etas = parse_float_list(ARGS[i])
        elseif arg == "--pilot"
            i += 1
            i <= length(ARGS) || error("Missing value after --pilot")
            opts.pilot_frac = parse(Float64, ARGS[i])
        elseif arg == "--pll-mode"
            i += 1
            i <= length(ARGS) || error("Missing value after --pll-mode")
            opts.pll_mode = lowercase(ARGS[i])
            opts.pll_mode in ("fixed", "learned", "joint") ||
                error("--pll-mode must be 'fixed', 'learned', or 'joint'")
        elseif arg == "--oracle-support"
            i += 1
            i <= length(ARGS) || error("Missing value after --oracle-support")
            opts.oracle_support = parse(Int, ARGS[i])
        elseif arg == "--start"
            i += 1
            i <= length(ARGS) || error("Missing value after --start")
            opts.packet_start = parse(Int, ARGS[i])
        elseif arg == "--end"
            i += 1
            i <= length(ARGS) || error("Missing value after --end")
            opts.packet_end = parse(Int, ARGS[i])
        elseif arg == "--chunk-size"
            i += 1
            i <= length(ARGS) || error("Missing value after --chunk-size")
            opts.chunk_size = parse(Int, ARGS[i])
        elseif arg == "--output-dir"
            i += 1
            i <= length(ARGS) || error("Missing value after --output-dir")
            opts.output_dir = isabspath(ARGS[i]) ? ARGS[i] : joinpath(pwd(), ARGS[i])
            opts.summary_csv = joinpath(opts.output_dir, "dfec_hparam_sweep_summary.csv")
        elseif arg == "--resume"
            opts.resume = true
        elseif startswith(arg, "--")
            error("Unknown option: $(arg)")
        else
            error("Unexpected positional argument: $(arg)")
        end
        i += 1
    end

    opts.oracle_support > 0 || error("--oracle-support must be positive")
    opts.chunk_size > 0 || error("--chunk-size must be positive")
    1 <= opts.packet_start <= opts.packet_end || error("Packet range must satisfy 1 <= start <= end")
    mkpath(opts.output_dir)
    return opts
end

function output_csv_for(opts::SweepOptions, λ::Float64, γ::Float64, η::Float64)
    pilot_label = @sprintf("%.2f", opts.pilot_frac)
    fname = "ldpc_ber_chunk1_pilot_$(pilot_label)_support_$(opts.oracle_support)_" *
            "lambda_$(fmt_float(λ))_gamma_$(fmt_float(γ))_eta_$(fmt_float(η))_$(opts.pll_mode).csv"
    return joinpath(opts.output_dir, fname)
end

function run_combo(project_root::String, runner::String, opts::SweepOptions, λ::Float64, γ::Float64, η::Float64)
    output_csv = output_csv_for(opts, λ, γ, η)
    julia_bin = joinpath(Sys.BINDIR, Base.julia_exename())

    cmd = Cmd([
        julia_bin,
        "--project=$(project_root)",
        runner,
        "--pll-mode", opts.pll_mode,
        "--pilot", string(opts.pilot_frac),
        "--oracle-support", string(opts.oracle_support),
        "--lambda", string(λ),
        "--gamma", string(γ),
        "--eta", string(η),
        "--start", string(opts.packet_start),
        "--end", string(opts.packet_end),
        "--chunk-size", string(opts.chunk_size),
        "--output", output_csv,
    ])

    if opts.resume
        cmd = Cmd(vcat(collect(cmd.exec), ["--resume"]))
    end

    println("\n" * "="^88)
    println("Chunk-1 DFEC hyperparameter sweep run")
    println("="^88)
    println("λ=$(λ) γ=$(γ) η=$(η)")
    println("Support:     $(opts.oracle_support)")
    println("Pilot frac:  $(opts.pilot_frac)")
    println("PLL mode:    $(opts.pll_mode)")
    println("Packet rows: $(opts.packet_start):$(opts.packet_end)")
    println("Output CSV:  $(output_csv)")
    println("="^88)

    run(cmd)
    return output_csv
end

function summarize_output(csv_path::String, λ::Float64, γ::Float64, η::Float64)
    results = CSV.read(csv_path, DataFrame)
    return (
        lambda=λ,
        gamma=γ,
        eta=η,
        rows=nrow(results),
        mean_dfec=mean(results.ber_dfec),
        mean_dfec_bp=mean(results.ber_dfec_bp),
    )
end

function main()
    project_root = dirname(@__DIR__)
    runner = joinpath(project_root, "scripts", "run_phase_corrected_900_dfec_bp.jl")
    isfile(runner) || error("Runner script not found: $(runner)")

    opts = parse_args(project_root)
    summary = DataFrame(
        lambda=Float64[],
        gamma=Float64[],
        eta=Float64[],
        rows=Int[],
        mean_dfec=Float64[],
        mean_dfec_bp=Float64[],
    )

    println("Lambda grid:  $(opts.lambdas)")
    println("Gamma grid:   $(opts.gammas)")
    println("Eta grid:     $(opts.etas)")
    println("Support:      $(opts.oracle_support)")
    println("Pilot frac:   $(opts.pilot_frac)")
    println("PLL mode:     $(opts.pll_mode)")
    println("Packet rows:  $(opts.packet_start):$(opts.packet_end)")
    println("Output dir:   $(opts.output_dir)")

    for λ in opts.lambdas
        for γ in opts.gammas
            for η in opts.etas
                csv_path = run_combo(project_root, runner, opts, λ, γ, η)
                push!(summary, summarize_output(csv_path, λ, γ, η))
                sort!(summary, [:mean_dfec_bp, :mean_dfec, :lambda, :gamma, :eta])
                CSV.write(opts.summary_csv, summary)
            end
        end
    end

    println("\nSummary")
    println(summary)
    best_idx = argmin(summary.mean_dfec_bp)
    println("\nBest DFEC hyperparameters by DFEC+BP BER:")
    println("λ=$(summary.lambda[best_idx]) γ=$(summary.gamma[best_idx]) η=$(summary.eta[best_idx])")
    println("Mean DFEC BER: $(round(summary.mean_dfec[best_idx], digits=6))")
    println("Mean DFEC+BP BER: $(round(summary.mean_dfec_bp[best_idx], digits=6))")
    println("Summary CSV: $(opts.summary_csv)")
end

main()
