"""
    run_phase_corrected_900_pilot_sweep.jl

Run the 900-packet phase-corrected DFEC+BP pipeline over multiple pilot ratios.
This is a thin wrapper around `run_phase_corrected_900_dfec_bp.jl` that launches
one full run per pilot fraction and writes one CSV per setting.

Defaults:
  - Pilot fractions: 0.36, 0.41, 0.46
  - PLL mode: joint
  - Packet rows: 1:900
  - Chunk size: 25
  - Output dir: results/

Usage:

    julia --project=. scripts/run_phase_corrected_900_pilot_sweep.jl

Resume a partial sweep:

    julia --project=. scripts/run_phase_corrected_900_pilot_sweep.jl --resume

Specify pilot fractions explicitly:

    julia --project=. scripts/run_phase_corrected_900_pilot_sweep.jl --pilots 0.36,0.41,0.46

Run a shorter range for debugging:

    julia --project=. scripts/run_phase_corrected_900_pilot_sweep.jl --pilots 0.41 --end 100
"""

using Printf

Base.@kwdef mutable struct SweepOptions
    pilots::Vector{Float64} = [0.36, 0.41, 0.46]
    pll_mode::String = "joint"
    chunk_size::Int = 25
    packet_start::Int = 1
    packet_end::Int = 900
    resume::Bool = false
    output_dir::String = ""
end

function parse_pilots(spec::String)
    vals = Float64[]
    for item in split(spec, ',')
        stripped = strip(item)
        isempty(stripped) && continue
        push!(vals, parse(Float64, stripped))
    end
    isempty(vals) && error("No pilot fractions parsed from --pilots")
    return vals
end

function pilot_label(pilot::Float64)
    return @sprintf("%.2f", pilot)
end

function mode_suffix(pll_mode::String)
    pll_mode == "joint" && return "jointpll"
    pll_mode == "learned" && return "plllearn"
    pll_mode == "fixed" && return "pll"
    error("Unsupported pll mode: $(pll_mode)")
end

function parse_args(project_root::String)
    opts = SweepOptions(output_dir=joinpath(project_root, "results"))

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--pilots"
            i += 1
            i <= length(ARGS) || error("Missing value after --pilots")
            opts.pilots = parse_pilots(ARGS[i])
        elseif arg == "--pll-mode"
            i += 1
            i <= length(ARGS) || error("Missing value after --pll-mode")
            opts.pll_mode = lowercase(ARGS[i])
            opts.pll_mode in ("fixed", "learned", "joint") ||
                error("--pll-mode must be 'fixed', 'learned', or 'joint'")
        elseif arg == "--chunk-size"
            i += 1
            i <= length(ARGS) || error("Missing value after --chunk-size")
            opts.chunk_size = parse(Int, ARGS[i])
        elseif arg == "--start"
            i += 1
            i <= length(ARGS) || error("Missing value after --start")
            opts.packet_start = parse(Int, ARGS[i])
        elseif arg == "--end"
            i += 1
            i <= length(ARGS) || error("Missing value after --end")
            opts.packet_end = parse(Int, ARGS[i])
        elseif arg == "--output-dir"
            i += 1
            i <= length(ARGS) || error("Missing value after --output-dir")
            opts.output_dir = isabspath(ARGS[i]) ? ARGS[i] : joinpath(pwd(), ARGS[i])
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

function output_csv_for(opts::SweepOptions, pilot::Float64)
    fname = "ldpc_ber_phase_corrected_900_pilot_$(pilot_label(pilot))_$(mode_suffix(opts.pll_mode)).csv"
    return joinpath(opts.output_dir, fname)
end

function run_one(project_root::String, runner::String, opts::SweepOptions, pilot::Float64)
    output_csv = output_csv_for(opts, pilot)
    julia_bin = joinpath(Sys.BINDIR, Base.julia_exename())

    cmd = Cmd([
        julia_bin,
        "--project=$(project_root)",
        runner,
        "--pll-mode", opts.pll_mode,
        "--pilot", string(pilot),
        "--start", string(opts.packet_start),
        "--end", string(opts.packet_end),
        "--chunk-size", string(opts.chunk_size),
        "--output", output_csv,
    ])

    if opts.resume
        cmd = Cmd(vcat(collect(cmd.exec), ["--resume"]))
    end

    println("\n" * "="^88)
    println("Pilot sweep run")
    println("="^88)
    println("Pilot frac:  $(pilot)")
    println("PLL mode:    $(opts.pll_mode)")
    println("Packet rows: $(opts.packet_start):$(opts.packet_end)")
    println("Chunk size:  $(opts.chunk_size)")
    println("Resume:      $(opts.resume)")
    println("Output CSV:  $(output_csv)")
    println("="^88)

    run(cmd)
end

function main()
    project_root = dirname(@__DIR__)
    runner = joinpath(project_root, "scripts", "run_phase_corrected_900_dfec_bp.jl")
    isfile(runner) || error("Runner script not found: $(runner)")

    opts = parse_args(project_root)

    println("Pilot ratios: $(opts.pilots)")
    println("Output dir:   $(opts.output_dir)")

    for pilot in opts.pilots
        run_one(project_root, runner, opts, pilot)
    end

    println("\n✅ Sweep finished")
    for pilot in opts.pilots
        println("   $(pilot): $(output_csv_for(opts, pilot))")
    end
end

main()
