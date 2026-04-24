"""
    sweep_first_chunk_channel_support.jl

Sweep the oracle channel support size over the first chunk (packet rows 1:25)
using the phase-corrected DFEC+BP pipeline.

Defaults:
  - Support sizes: 2, 4, 6, 8, 10, 12, 14, 16
  - Pilot frac: 0.41
  - PLL mode: joint
  - Packet rows: 1:25
  - Chunk size: 25

Usage:

    julia --project=. scripts/sweep_first_chunk_channel_support.jl

Resume partial runs:

    julia --project=. scripts/sweep_first_chunk_channel_support.jl --resume

Specify support sizes explicitly:

    julia --project=. scripts/sweep_first_chunk_channel_support.jl --supports 4,6,8,10,12
"""

using Printf
using CSV, DataFrames, Statistics

Base.@kwdef mutable struct SweepOptions
    supports::Vector{Int} = collect(2:2:16)
    pilot_frac::Float64 = 0.41
    pll_mode::String = "joint"
    packet_start::Int = 1
    packet_end::Int = 25
    chunk_size::Int = 25
    resume::Bool = false
    output_dir::String = ""
    summary_csv::String = ""
end

function parse_supports(spec::String)
    vals = Int[]
    for item in split(spec, ',')
        stripped = strip(item)
        isempty(stripped) && continue
        push!(vals, parse(Int, stripped))
    end
    isempty(vals) && error("No support sizes parsed from --supports")
    return vals
end

function parse_args(project_root::String)
    outdir = joinpath(project_root, "results", "support_sweep_chunk1")
    opts = SweepOptions(output_dir=outdir, summary_csv=joinpath(outdir, "support_sweep_summary.csv"))

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--supports"
            i += 1
            i <= length(ARGS) || error("Missing value after --supports")
            opts.supports = parse_supports(ARGS[i])
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
            opts.summary_csv = joinpath(opts.output_dir, "support_sweep_summary.csv")
        elseif arg == "--resume"
            opts.resume = true
        elseif startswith(arg, "--")
            error("Unknown option: $(arg)")
        else
            error("Unexpected positional argument: $(arg)")
        end
        i += 1
    end

    opts.chunk_size > 0 || error("--chunk-size must be positive")
    1 <= opts.packet_start <= opts.packet_end || error("Packet range must satisfy 1 <= start <= end")
    mkpath(opts.output_dir)
    return opts
end

function output_csv_for(opts::SweepOptions, support::Int)
    pilot_label = @sprintf("%.2f", opts.pilot_frac)
    fname = "ldpc_ber_chunk1_pilot_$(pilot_label)_support_$(support)_$(opts.pll_mode).csv"
    return joinpath(opts.output_dir, fname)
end

function run_support(project_root::String, runner::String, opts::SweepOptions, support::Int)
    output_csv = output_csv_for(opts, support)
    julia_bin = joinpath(Sys.BINDIR, Base.julia_exename())

    cmd = Cmd([
        julia_bin,
        "--project=$(project_root)",
        runner,
        "--pll-mode", opts.pll_mode,
        "--pilot", string(opts.pilot_frac),
        "--oracle-support", string(support),
        "--start", string(opts.packet_start),
        "--end", string(opts.packet_end),
        "--chunk-size", string(opts.chunk_size),
        "--output", output_csv,
    ])

    if opts.resume
        cmd = Cmd(vcat(collect(cmd.exec), ["--resume"]))
    end

    println("\n" * "="^88)
    println("Chunk-1 support sweep run")
    println("="^88)
    println("Support:     $(support)")
    println("Pilot frac:  $(opts.pilot_frac)")
    println("PLL mode:    $(opts.pll_mode)")
    println("Packet rows: $(opts.packet_start):$(opts.packet_end)")
    println("Output CSV:  $(output_csv)")
    println("="^88)

    run(cmd)
    return output_csv
end

function summarize_output(csv_path::String, support::Int)
    results = CSV.read(csv_path, DataFrame)
    return (
        oracle_support=support,
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
        oracle_support=Int[],
        rows=Int[],
        mean_dfec=Float64[],
        mean_dfec_bp=Float64[],
    )

    println("Support sizes: $(opts.supports)")
    println("Pilot frac:    $(opts.pilot_frac)")
    println("PLL mode:      $(opts.pll_mode)")
    println("Packet rows:   $(opts.packet_start):$(opts.packet_end)")
    println("Output dir:    $(opts.output_dir)")

    for support in opts.supports
        csv_path = run_support(project_root, runner, opts, support)
        push!(summary, summarize_output(csv_path, support))
        sort!(summary, :oracle_support)
        CSV.write(opts.summary_csv, summary)
    end

    println("\nSummary")
    println(summary)
    best_idx = argmin(summary.mean_dfec_bp)
    println("\nBest support by DFEC+BP BER: $(summary.oracle_support[best_idx])")
    println("Mean DFEC BER: $(round(summary.mean_dfec[best_idx], digits=6))")
    println("Mean DFEC+BP BER: $(round(summary.mean_dfec_bp[best_idx], digits=6))")
    println("Summary CSV: $(opts.summary_csv)")
end

main()
