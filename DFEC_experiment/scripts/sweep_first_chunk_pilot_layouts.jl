"""
    sweep_first_chunk_pilot_layouts.jl

Sweep pilot-position layouts over the first chunk (packet rows 1:25)
using the current best DFEC settings.

Defaults:
  - Layouts: tailrows, headrows, midrows, spreadrows, front, center, uniform, random
  - Pilot frac: 0.41
  - PLL mode: joint
  - Oracle support: 8
  - DFEC params: λ=2.0, γ=1e-3, η=1.0
  - Packet rows: 1:25

Usage:

    julia --project=. scripts/sweep_first_chunk_pilot_layouts.jl

Resume partial runs:

    julia --project=. scripts/sweep_first_chunk_pilot_layouts.jl --resume
"""

using Printf
using CSV, DataFrames, Statistics

Base.@kwdef mutable struct SweepOptions
    layouts::Vector{String} = ["tailrows", "headrows", "midrows", "spreadrows", "front", "center", "uniform", "random"]
    pilot_frac::Float64 = 0.41
    pll_mode::String = "joint"
    oracle_support::Int = 8
    lambda::Float64 = 2.0
    gamma::Float64 = 1e-3
    eta::Float64 = 1.0
    packet_start::Int = 1
    packet_end::Int = 25
    chunk_size::Int = 25
    resume::Bool = false
    output_dir::String = ""
    summary_csv::String = ""
end

function parse_string_list(spec::String)
    vals = String[]
    for item in split(spec, ',')
        stripped = lowercase(strip(item))
        isempty(stripped) && continue
        push!(vals, stripped)
    end
    isempty(vals) && error("No values parsed")
    return vals
end

function parse_args(project_root::String)
    outdir = joinpath(project_root, "results", "pilot_layout_sweep_chunk1")
    opts = SweepOptions(output_dir=outdir, summary_csv=joinpath(outdir, "pilot_layout_sweep_summary.csv"))

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--layouts"
            i += 1
            i <= length(ARGS) || error("Missing value after --layouts")
            opts.layouts = parse_string_list(ARGS[i])
        elseif arg == "--pilot"
            i += 1
            i <= length(ARGS) || error("Missing value after --pilot")
            opts.pilot_frac = parse(Float64, ARGS[i])
        elseif arg == "--pll-mode"
            i += 1
            i <= length(ARGS) || error("Missing value after --pll-mode")
            opts.pll_mode = lowercase(ARGS[i])
        elseif arg == "--oracle-support"
            i += 1
            i <= length(ARGS) || error("Missing value after --oracle-support")
            opts.oracle_support = parse(Int, ARGS[i])
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
            opts.summary_csv = joinpath(opts.output_dir, "pilot_layout_sweep_summary.csv")
        elseif arg == "--resume"
            opts.resume = true
        elseif startswith(arg, "--")
            error("Unknown option: $(arg)")
        else
            error("Unexpected positional argument: $(arg)")
        end
        i += 1
    end

    mkpath(opts.output_dir)
    return opts
end

function output_csv_for(opts::SweepOptions, layout::String)
    pilot_label = @sprintf("%.2f", opts.pilot_frac)
    fname = "ldpc_ber_chunk1_pilot_$(pilot_label)_layout_$(layout)_support_$(opts.oracle_support)_$(opts.pll_mode).csv"
    return joinpath(opts.output_dir, fname)
end

function run_one(project_root::String, runner::String, opts::SweepOptions, layout::String)
    output_csv = output_csv_for(opts, layout)
    julia_bin = joinpath(Sys.BINDIR, Base.julia_exename())
    cmd = Cmd([
        julia_bin,
        "--project=$(project_root)",
        "--startup-file=no",
        runner,
        "--pilot", string(opts.pilot_frac),
        "--pll-mode", opts.pll_mode,
        "--pilot-layout", layout,
        "--oracle-support", string(opts.oracle_support),
        "--lambda", string(opts.lambda),
        "--gamma", string(opts.gamma),
        "--eta", string(opts.eta),
        "--start", string(opts.packet_start),
        "--end", string(opts.packet_end),
        "--chunk-size", string(opts.chunk_size),
        "--output", output_csv,
    ])
    opts.resume && (cmd = `$cmd --resume`)

    println("\n" * "="^88)
    println("Chunk-1 pilot-layout sweep run")
    println("="^88)
    println("Layout:      $(layout)")
    println("Pilot frac:  $(opts.pilot_frac)")
    println("PLL mode:    $(opts.pll_mode)")
    println("Support:     $(opts.oracle_support)")
    println("DFEC params: λ=$(opts.lambda) γ=$(opts.gamma) η=$(opts.eta)")
    println("Packet rows: $(opts.packet_start):$(opts.packet_end)")
    println("Output CSV:  $(output_csv)")
    println("="^88)
    run(cmd)
    return output_csv
end

function summarize_layout(csv_path::String)
    df = CSV.read(csv_path, DataFrame)
    return (
        rows=nrow(df),
        mean_dfec=mean(df.ber_dfec),
        mean_dfec_bp=mean(df.ber_dfec_bp),
    )
end

function main()
    project_root = dirname(@__DIR__)
    runner = joinpath(project_root, "scripts", "run_phase_corrected_900_dfec_bp.jl")
    opts = parse_args(project_root)

    println("="^88)
    println("Chunk-1 pilot layout sweep")
    println("="^88)
    println("Layouts:     $(opts.layouts)")
    println("Pilot frac:  $(opts.pilot_frac)")
    println("PLL mode:    $(opts.pll_mode)")
    println("Support:     $(opts.oracle_support)")
    println("DFEC params: λ=$(opts.lambda) γ=$(opts.gamma) η=$(opts.eta)")
    println("Packet rows: $(opts.packet_start):$(opts.packet_end)")
    println("Output dir:  $(opts.output_dir)")
    println("="^88)

    summary = DataFrame(
        pilot_layout=String[],
        rows=Int[],
        mean_dfec=Float64[],
        mean_dfec_bp=Float64[],
    )

    for layout in opts.layouts
        csv_path = run_one(project_root, runner, opts, layout)
        stats = summarize_layout(csv_path)
        push!(summary, (
            pilot_layout=layout,
            rows=stats.rows,
            mean_dfec=stats.mean_dfec,
            mean_dfec_bp=stats.mean_dfec_bp,
        ))
        sort!(summary, :mean_dfec_bp)
        CSV.write(opts.summary_csv, summary)
    end

    sort!(summary, :mean_dfec_bp)
    CSV.write(opts.summary_csv, summary)

    println("\nSummary")
    show(stdout, MIME"text/plain"(), summary)
    println("\n\nBest pilot layout by DFEC+BP BER:")
    println(summary[1, :pilot_layout])
    @printf("Mean DFEC BER: %.6f\n", summary[1, :mean_dfec])
    @printf("Mean DFEC+BP BER: %.6f\n", summary[1, :mean_dfec_bp])
    println("Summary CSV: $(opts.summary_csv)")
end

main()
