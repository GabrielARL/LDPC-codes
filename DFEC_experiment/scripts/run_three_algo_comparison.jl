"""
    run_three_algo_comparison.jl

Run `DFE+FEC`, `Turbo EQ`, and `DFEC+BP` on the same packet set with the same
packet preprocessing path, then write a compact comparison summary.

This is a thin wrapper around `run_phase_corrected_900_dfec_bp.jl`, which
already computes all three BER columns from one shared per-packet pipeline.
The wrapper exists to make the comparison explicit and repeatable.

Comparable BER columns in the detailed CSV:
  - `ber_dfe_fec`
  - `ber_turbo`      (`Turbo EQ` symbol/codeword BER)
  - `ber_dfec_bp`

Note:
  This script intentionally reports the unified apples-to-apples comparison.
  It does not try to reproduce the legacy standalone `exp_tb_process.jl`
  summary, which used a different turbo BER definition (`ber_turbo_info`)
  and, in earlier discussions, was sometimes quoted on a smaller subset.

Defaults:
  - Pilot frac:     0.35
  - Pilot layout:   uniform
  - PLL mode:       joint
  - Turbo source:   shared
  - Oracle support: 8
  - DFEC params:    lambda=2.0 gamma=1e-3 eta=1.0
  - Packet rows:    1:900
  - Chunk size:     25

Usage:

    julia --project=. scripts/run_three_algo_comparison.jl

Run a shorter debug subset:

    julia --project=. scripts/run_three_algo_comparison.jl --start 1 --end 25

Resume a partial comparison:

    julia --project=. scripts/run_three_algo_comparison.jl --resume
"""

using CSV, DataFrames, Statistics, Printf

const DEFAULT_PILOT_LAYOUT = "uniform"
const PILOT_LAYOUT_CHOICES = ("tailrows", "headrows", "midrows", "spreadrows", "front", "center", "uniform", "random")
const DEFAULT_PLL_MODE = "joint"
const PLL_MODE_CHOICES = ("fixed", "learned", "joint")
const DEFAULT_TURBO_SOURCE = "shared"
const TURBO_SOURCE_CHOICES = ("shared", "legacy-csv")

Base.@kwdef mutable struct CompareOptions
    pilot_frac::Float64 = 0.35
    pilot_layout::String = DEFAULT_PILOT_LAYOUT
    pll_mode::String = DEFAULT_PLL_MODE
    turbo_source::String = DEFAULT_TURBO_SOURCE
    legacy_turbo_csv::String = ""
    oracle_support::Int = 8
    lambda::Float64 = 2.0
    gamma::Float64 = 1e-3
    eta::Float64 = 1.0
    packet_start::Int = 1
    packet_end::Int = 900
    chunk_size::Int = 25
    resume::Bool = false
    output_dir::String = ""
    output_csv::String = ""
    summary_csv::String = ""
end

function pilot_label(x::Float64)
    return @sprintf("%.2f", x)
end

function fmt_float(x::Float64)
    s = @sprintf("%.6g", x)
    return replace(s, "." => "p", "-" => "m", "+" => "")
end

function default_stem(opts::CompareOptions)
    return "three_algo_comparison_pilot_$(pilot_label(opts.pilot_frac))_" *
           "layout_$(opts.pilot_layout)_$(opts.pll_mode)_" *
           "support_$(opts.oracle_support)_" *
           "lambda_$(fmt_float(opts.lambda))_" *
           "gamma_$(fmt_float(opts.gamma))_" *
           "eta_$(fmt_float(opts.eta))"
end

function resolve_output_paths!(opts::CompareOptions, project_root::String)
    if isempty(opts.output_dir)
        opts.output_dir = joinpath(project_root, "results", "three_algo_comparison")
    elseif !isabspath(opts.output_dir)
        opts.output_dir = joinpath(pwd(), opts.output_dir)
    end
    mkpath(opts.output_dir)

    stem = default_stem(opts)
    if isempty(opts.output_csv)
        opts.output_csv = joinpath(opts.output_dir, stem * ".csv")
    elseif !isabspath(opts.output_csv)
        opts.output_csv = joinpath(pwd(), opts.output_csv)
    end

    if isempty(opts.summary_csv)
        opts.summary_csv = joinpath(opts.output_dir, stem * "_summary.csv")
    elseif !isabspath(opts.summary_csv)
        opts.summary_csv = joinpath(pwd(), opts.summary_csv)
    end
end

function parse_args(project_root::String)
    opts = CompareOptions()

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--pilot"
            i += 1
            i <= length(ARGS) || error("Missing value after --pilot")
            opts.pilot_frac = parse(Float64, ARGS[i])
        elseif arg == "--pilot-layout"
            i += 1
            i <= length(ARGS) || error("Missing value after --pilot-layout")
            opts.pilot_layout = lowercase(ARGS[i])
        elseif arg == "--pll-mode"
            i += 1
            i <= length(ARGS) || error("Missing value after --pll-mode")
            opts.pll_mode = lowercase(ARGS[i])
        elseif arg == "--turbo-source"
            i += 1
            i <= length(ARGS) || error("Missing value after --turbo-source")
            opts.turbo_source = lowercase(ARGS[i])
        elseif arg == "--legacy-turbo-csv"
            i += 1
            i <= length(ARGS) || error("Missing value after --legacy-turbo-csv")
            opts.legacy_turbo_csv = ARGS[i]
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
            opts.output_dir = ARGS[i]
        elseif arg == "--output"
            i += 1
            i <= length(ARGS) || error("Missing value after --output")
            opts.output_csv = ARGS[i]
        elseif arg == "--summary"
            i += 1
            i <= length(ARGS) || error("Missing value after --summary")
            opts.summary_csv = ARGS[i]
        elseif arg == "--resume"
            opts.resume = true
        elseif startswith(arg, "--")
            error("Unknown option: $(arg)")
        else
            error("Unexpected positional argument: $(arg)")
        end
        i += 1
    end

    opts.pilot_layout in PILOT_LAYOUT_CHOICES ||
        error("--pilot-layout must be one of $(join(PILOT_LAYOUT_CHOICES, ", "))")
    opts.pll_mode in PLL_MODE_CHOICES ||
        error("--pll-mode must be one of $(join(PLL_MODE_CHOICES, ", "))")
    opts.turbo_source in TURBO_SOURCE_CHOICES ||
        error("--turbo-source must be one of $(join(TURBO_SOURCE_CHOICES, ", "))")
    opts.oracle_support > 0 || error("--oracle-support must be positive")
    opts.lambda > 0 || error("--lambda must be positive")
    opts.gamma > 0 || error("--gamma must be positive")
    opts.eta >= 0 || error("--eta must be nonnegative")
    opts.chunk_size > 0 || error("--chunk-size must be positive")
    1 <= opts.packet_start <= opts.packet_end || error("Packet range must satisfy 1 <= start <= end")

    resolve_output_paths!(opts, project_root)
    if opts.turbo_source == "legacy-csv" && isempty(opts.legacy_turbo_csv)
        opts.legacy_turbo_csv = joinpath(dirname(project_root), "Process MF 1.6km exp", "tb_ber.csv")
    elseif !isempty(opts.legacy_turbo_csv) && !isabspath(opts.legacy_turbo_csv)
        opts.legacy_turbo_csv = joinpath(pwd(), opts.legacy_turbo_csv)
    end
    return opts
end

function build_runner_cmd(project_root::String, runner::String, opts::CompareOptions)
    julia_bin = joinpath(Sys.BINDIR, Base.julia_exename())
    args = [
        julia_bin,
        "--project=$(project_root)",
        runner,
        "--pll-mode", opts.pll_mode,
        "--pilot", string(opts.pilot_frac),
        "--pilot-layout", opts.pilot_layout,
        "--oracle-support", string(opts.oracle_support),
        "--lambda", string(opts.lambda),
        "--gamma", string(opts.gamma),
        "--eta", string(opts.eta),
        "--start", string(opts.packet_start),
        "--end", string(opts.packet_end),
        "--chunk-size", string(opts.chunk_size),
        "--output", opts.output_csv,
    ]
    opts.resume && append!(args, ["--resume"])
    return Cmd(args)
end

function summarize_results(results::DataFrame)
    metric_cols = [
        ("DFE+FEC", :ber_dfe_fec),
        ("Turbo EQ (symbol BER)", :ber_turbo),
        ("DFEC+BP", :ber_dfec_bp),
    ]

    summary = DataFrame(
        algorithm=String[],
        metric_column=String[],
        rows=Int[],
        mean_ber=Float64[],
        std_ber=Float64[],
        median_ber=Float64[],
        min_ber=Float64[],
        max_ber=Float64[],
        best_rate=Float64[],
    )

    n = nrow(results)
    best_per_row = [minimum((results.ber_dfe_fec[i], results.ber_turbo[i], results.ber_dfec_bp[i])) for i in 1:n]

    for (label, col) in metric_cols
        vals = Vector{Float64}(results[!, col])
        best_hits = count(i -> abs(vals[i] - best_per_row[i]) <= 1e-12, eachindex(vals))
        push!(summary, (
            algorithm=label,
            metric_column=String(col),
            rows=length(vals),
            mean_ber=mean(vals),
            std_ber=std(vals),
            median_ber=median(vals),
            min_ber=minimum(vals),
            max_ber=maximum(vals),
            best_rate=best_hits / length(vals),
        ))
    end

    sort!(summary, [:mean_ber, :algorithm])
    return summary
end

function nearest_pilot_value(pilot_fracs::AbstractVector{<:Real}, target::Float64)
    best_idx = 1
    best_dist = Inf
    for (i, val) in pairs(pilot_fracs)
        dist = abs(Float64(val) - target)
        if dist < best_dist
            best_dist = dist
            best_idx = i
        end
    end
    return Float64(pilot_fracs[best_idx])
end

function load_legacy_turbo_stats(opts::CompareOptions)
    isfile(opts.legacy_turbo_csv) || error("Legacy turbo CSV not found: $(opts.legacy_turbo_csv)")
    df = CSV.read(opts.legacy_turbo_csv, DataFrame)
    :pilot_frac in propertynames(df) || error("Legacy turbo CSV must contain a pilot_frac column")

    if :ber_tb in propertynames(df)
        available = unique(Float64.(df.pilot_frac))
        chosen = nearest_pilot_value(available, opts.pilot_frac)
        sel = df[abs.(Float64.(df.pilot_frac) .- chosen) .<= 1e-12, :]
        vals = Float64.(sel.ber_tb)
        return (
            source="legacy-csv",
            metric_column="ber_tb",
            effective_pilot_frac=chosen,
            rows=length(vals),
            mean_ber=mean(vals),
            std_ber=std(vals),
            median_ber=median(vals),
            min_ber=minimum(vals),
            max_ber=maximum(vals),
            path=opts.legacy_turbo_csv,
        )
    elseif :avg_tb_ber in propertynames(df)
        available = unique(Float64.(df.pilot_frac))
        chosen = nearest_pilot_value(available, opts.pilot_frac)
        sel = df[abs.(Float64.(df.pilot_frac) .- chosen) .<= 1e-12, :]
        nrow(sel) == 0 && error("No legacy turbo rows found near pilot_frac=$(opts.pilot_frac)")
        val = Float64(sel.avg_tb_ber[1])
        std_val = :std_tb_ber in propertynames(sel) ? Float64(sel.std_tb_ber[1]) : NaN
        return (
            source="legacy-csv",
            metric_column="avg_tb_ber",
            effective_pilot_frac=chosen,
            rows=1,
            mean_ber=val,
            std_ber=std_val,
            median_ber=val,
            min_ber=val,
            max_ber=val,
            path=opts.legacy_turbo_csv,
        )
    else
        error("Legacy turbo CSV must contain either ber_tb or avg_tb_ber")
    end
end

function summarize_results_legacy_turbo(results::DataFrame, opts::CompareOptions)
    summary = DataFrame(
        algorithm=String[],
        metric_column=String[],
        source=String[],
        rows=Int[],
        effective_pilot_frac=Float64[],
        mean_ber=Float64[],
        std_ber=Float64[],
        median_ber=Float64[],
        min_ber=Float64[],
        max_ber=Float64[],
    )

    for (label, col) in (("DFE+FEC", :ber_dfe_fec), ("DFEC+BP", :ber_dfec_bp))
        vals = Vector{Float64}(results[!, col])
        push!(summary, (
            algorithm=label,
            metric_column=String(col),
            source="shared-runner",
            rows=length(vals),
            effective_pilot_frac=opts.pilot_frac,
            mean_ber=mean(vals),
            std_ber=std(vals),
            median_ber=median(vals),
            min_ber=minimum(vals),
            max_ber=maximum(vals),
        ))
    end

    turbo = load_legacy_turbo_stats(opts)
    push!(summary, (
        algorithm="Turbo EQ (legacy)",
        metric_column=turbo.metric_column,
        source=turbo.source,
        rows=turbo.rows,
        effective_pilot_frac=turbo.effective_pilot_frac,
        mean_ber=turbo.mean_ber,
        std_ber=turbo.std_ber,
        median_ber=turbo.median_ber,
        min_ber=turbo.min_ber,
        max_ber=turbo.max_ber,
    ))

    sort!(summary, [:mean_ber, :algorithm])
    return summary
end

function print_summary(summary::DataFrame, opts::CompareOptions, rows_written::Int)
    if opts.turbo_source == "shared"
        println("\nUnified Comparable BER Summary")
    else
        println("\nMixed Summary With Legacy Turbo")
    end
    println("="^88)
    println(
        "Scope: rows $(opts.packet_start):$(opts.packet_end), " *
        "pilot=$(opts.pilot_frac), layout=$(opts.pilot_layout), pll=$(opts.pll_mode), " *
        "support=$(opts.oracle_support), rows_written=$(rows_written)"
    )
    if opts.turbo_source == "shared"
        println(
            "Metrics: DFE+FEC BER, Turbo EQ symbol BER, DFEC+BP BER " *
            "(same packet set, same preprocessing path)"
        )
    else
        println(
            "Metrics: DFE+FEC BER and DFEC+BP BER come from the shared runner; " *
            "Turbo uses the legacy standalone CSV"
        )
        println("Legacy turbo CSV: $(opts.legacy_turbo_csv)")
    end
    for row in eachrow(summary)
        line = rpad(row.algorithm, 24) *
               " mean=$(round(row.mean_ber, digits=6))" *
               " std=$(round(row.std_ber, digits=6))" *
               " median=$(round(row.median_ber, digits=6))"
        if :best_rate in propertynames(summary)
            line *= " best_rate=$(round(100 * row.best_rate, digits=2))%"
        else
            line *= " source=$(row.source) pilot_used=$(round(row.effective_pilot_frac, digits=4))"
        end
        println(line)
    end
    println("="^88)
    println("Best by mean BER: $(summary.algorithm[1]) ($(round(summary.mean_ber[1], digits=6)))")
end

function main()
    project_root = dirname(@__DIR__)
    runner = joinpath(project_root, "scripts", "run_phase_corrected_900_dfec_bp.jl")
    isfile(runner) || error("Runner script not found: $(runner)")

    opts = parse_args(project_root)

    println("="^88)
    println("THREE-ALGORITHM COMPARISON")
    println("="^88)
    println("Runner:        $(runner)")
    println("Pilot frac:    $(opts.pilot_frac)")
    println("Pilot layout:  $(opts.pilot_layout)")
    println("PLL mode:      $(opts.pll_mode)")
    println("Turbo source:  $(opts.turbo_source)")
    println("Support:       $(opts.oracle_support)")
    println("DFEC params:   lambda=$(opts.lambda) gamma=$(opts.gamma) eta=$(opts.eta)")
    println("Packet rows:   $(opts.packet_start):$(opts.packet_end)")
    println("Chunk size:    $(opts.chunk_size)")
    println("Resume mode:   $(opts.resume)")
    println("Detailed CSV:  $(opts.output_csv)")
    println("Summary CSV:   $(opts.summary_csv)")
    println("="^88)

    cmd = build_runner_cmd(project_root, runner, opts)
    run(cmd)

    results = CSV.read(opts.output_csv, DataFrame)
    summary = opts.turbo_source == "shared" ? summarize_results(results) : summarize_results_legacy_turbo(results, opts)
    mkpath(dirname(opts.summary_csv))
    CSV.write(opts.summary_csv, summary)

    print_summary(summary, opts, nrow(results))
    println("Detailed results: $(opts.output_csv)")
    println("Summary results:  $(opts.summary_csv)")
end

main()
