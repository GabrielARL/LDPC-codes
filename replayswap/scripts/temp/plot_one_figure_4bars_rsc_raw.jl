#!/usr/bin/env julia
# scripts/plot_groupbars_by_p_rsc_raw.jl
#
# Grouped bars: each pilot ratio p is a group on x-axis.
# Bars in each group:
#   - linear (RSC)  from psweep_replayswap_bpsk_RSC64_128_linear.csv
#   - turbo  (RSC)  from psweep_replayswap_bpsk_RSC64_128_turbo.csv
#   - eq+spa (RAW)  from compare_raw_ldpc_EQSPA_JSDC.csv (repeated across p)
#   - jsdc   (RAW)  from compare_raw_ldpc_EQSPA_JSDC.csv (repeated across p)
#
# Metric is BER (lower is better).
# RSC uses: b128_post_ber (default) or u64_ber (--metric u64_ber)
# RAW uses: ber_eqspa / ber_jsdc
#
# If any p in --p_grid is missing in either RSC csv, script errors and prints missing p.
#
# Usage:
#   julia --project=. scripts/plot_groupbars_by_p_rsc_raw.jl
#   julia --project=. scripts/plot_groupbars_by_p_rsc_raw.jl --metric u64_ber
#   julia --project=. scripts/plot_groupbars_by_p_rsc_raw.jl --agg median
#
using CSV, DataFrames, Statistics, Printf
using CairoMakie

die(msg::String) = error(msg)

parse_int(s::String) = parse(Int, strip(s))

function parse_floats(s::String)
    t = replace(strip(s), " " => "")
    isempty(t) && return Float64[]
    return Float64.(parse.(Float64, split(t, ",")))
end

function aggfun(name::String)
    n = lowercase(strip(name))
    n == "mean"   && return mean
    n == "median" && return median
    die("--agg must be mean or median")
end

function require_col(df::DataFrame, c::Symbol, path::String)
    hasproperty(df, c) || die("Missing column $(c) in $path. Columns: $(names(df))")
end

function missing_ps(df::DataFrame, p_grid::Vector{Float64})
    have = Set(round.(Float64.(df.p); digits=6))
    miss = Float64[]
    for p in p_grid
        want = round(p; digits=6)
        (want in have) || push!(miss, p)
    end
    return miss
end

# Returns Vector{Float64} of length |p_grid|:
#   v[p] = agg(metric_col over blocks at that p)
function rsc_series(psweep_csv::String; metric_col::Symbol, fagg, p_grid::Vector{Float64})
    isfile(psweep_csv) || die("Missing RSC psweep CSV: $psweep_csv")
    df = CSV.read(psweep_csv, DataFrame)

    println("RSC CSV: $psweep_csv")
    println("  rows=$(nrow(df)) cols=$(ncol(df))")
    println("  columns=", names(df))

    require_col(df, :p, psweep_csv)
    require_col(df, metric_col, psweep_csv)

    df.p = Float64.(df.p)
    df[!, metric_col] = Float64.(df[!, metric_col])

    miss = missing_ps(df, p_grid)
    if !isempty(miss)
        println("==============================================================")
        println("ERROR: Not enough data in: $psweep_csv")
        println("Required p grid: ", p_grid)
        println("Missing p values: ", miss)
        println("Available p values: ", sort(unique(df.p)))
        println("==============================================================")
        error("Missing required p values in RSC psweep CSV.")
    end

    vals = Float64[]
    for p in p_grid
        want = round(p; digits=6)
        sub = df[round.(df.p; digits=6) .== want, metric_col]
        isempty(sub) && error("Internal: p=$p unexpectedly empty after checks.")
        push!(vals, fagg(sub))
    end
    return vals
end

function raw_values(raw_csv::String; fagg)
    isfile(raw_csv) || die("Missing RAW CSV: $raw_csv")
    df = CSV.read(raw_csv, DataFrame)

    println("RAW CSV: $raw_csv")
    println("  rows=$(nrow(df)) cols=$(ncol(df))")
    println("  columns=", names(df))

    require_col(df, :ber_eqspa, raw_csv)
    require_col(df, :ber_jsdc, raw_csv)

    df.ber_eqspa = Float64.(df.ber_eqspa)
    df.ber_jsdc  = Float64.(df.ber_jsdc)

    return fagg(df.ber_eqspa), fagg(df.ber_jsdc)
end

function main()
    # defaults
    rsc_linear_csv = "data/runs/psweep_replayswap_bpsk_RSC64_128_linear.csv"
    rsc_turbo_csv  = "data/runs/psweep_replayswap_bpsk_RSC64_128_turbo.csv"
    raw_csv        = "data/runs/compare_raw_ldpc_EQSPA_JSDC.csv"

    out_png = "data/plots/groupbars_by_p_rsc_raw.png"
    out_pdf = "data/plots/groupbars_by_p_rsc_raw.pdf"

    metric = "b128_ber"   # b128_ber | u64_ber
    agg = "mean"          # mean|median
    p_grid = [0.0,0.1,0.2,0.3,0.4,0.5]

    # optionally hide linear (bars=3 => turbo,eq+spa,jsdc)
    bars = 4

    # CLI
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--metric"; i += 1; metric = ARGS[i]
        elseif a == "--agg"; i += 1; agg = ARGS[i]
        elseif a == "--p_grid"; i += 1; p_grid = parse_floats(ARGS[i])
        elseif a == "--bars"; i += 1; bars = parse_int(ARGS[i])
        elseif a == "--rsc_linear"; i += 1; rsc_linear_csv = ARGS[i]
        elseif a == "--rsc_turbo"; i += 1; rsc_turbo_csv = ARGS[i]
        elseif a == "--raw"; i += 1; raw_csv = ARGS[i]
        elseif a == "--out"; i += 1; out_png = ARGS[i]
        elseif a == "--outpdf"; i += 1; out_pdf = ARGS[i]
        elseif a == "--help" || a == "-h"
            println("""
Usage:
  julia --project=. scripts/plot_groupbars_by_p_rsc_raw.jl [args]

Args:
  --metric b128_ber|u64_ber     (default b128_ber)
  --agg mean|median            (default mean)
  --p_grid "0,0.1,0.2,0.3,0.4,0.5"
  --bars 4|3                   (default 4; 3 => turbo,eq+spa,jsdc)
  --out <png>
  --outpdf <pdf>

Notes:
  - RAW eq+spa/jsdc are constant across p (repeated bars).
""")
            return
        else
            die("Unknown arg: $a")
        end
        i += 1
    end

    (bars == 4 || bars == 3) || die("--bars must be 3 or 4")
    isempty(p_grid) && die("--p_grid cannot be empty")

    fagg = aggfun(agg)

    metric_col = lowercase(metric) == "b128_ber" ? :b128_post_ber :
                 lowercase(metric) == "u64_ber"  ? :u64_ber :
                 die("--metric must be b128_ber or u64_ber")

    # RSC series (vary with p)
    v_linear = Float64[]
    if bars == 4
        v_linear = rsc_series(rsc_linear_csv; metric_col=metric_col, fagg=fagg, p_grid=p_grid)
    end
    v_turbo  = rsc_series(rsc_turbo_csv;  metric_col=metric_col, fagg=fagg, p_grid=p_grid)

    # RAW constants (repeat for each p group)
    v_eqspa_const, v_jsdc_const = raw_values(raw_csv; fagg=fagg)
    v_eqspa = fill(v_eqspa_const, length(p_grid))
    v_jsdc  = fill(v_jsdc_const, length(p_grid))

    # build series list
    series_names = String[]
    series_vals  = Vector{Vector{Float64}}()
    series_cols  = Any[]

    if bars == 4
        push!(series_names, "linear"); push!(series_vals, v_linear); push!(series_cols, (:dodgerblue, 0.85))
    end
    push!(series_names, "turbo");  push!(series_vals, v_turbo);  push!(series_cols, (:seagreen, 0.85))
    push!(series_names, "eq+spa"); push!(series_vals, v_eqspa);  push!(series_cols, (:orange, 0.85))
    push!(series_names, "jsdc");   push!(series_vals, v_jsdc);   push!(series_cols, (:purple, 0.85))

    # Plot grouped bars
    mkpath(dirname(out_png))
    mkpath(dirname(out_pdf))

    title_metric = (metric_col == :b128_post_ber) ? "b128 BER" : "u64 BER"
    p_grid_str = join([@sprintf("%.1f", p) for p in p_grid], ",")
    title = "Grouped BER by pilot ratio p | $title_metric | p_grid=$p_grid_str"

    fig = Figure(size=(1200, 520))
    ax = Axis(fig[1,1],
        title=title,
        xlabel="Pilot ratio p",
        ylabel="BER (lower is better)"
    )

    # y-lims
    allvals = reduce(vcat, series_vals)
    ylims!(ax, 0.0, max(0.02, 1.15*maximum(allvals)))

    nG = length(p_grid)
    nS = length(series_names)
    xcenters = collect(1:nG)
    group_w = 0.85
    bar_w = group_w / nS

    for (k, (name, vals, col)) in enumerate(zip(series_names, series_vals, series_cols))
        offset = -group_w/2 + (k - 0.5) * bar_w
        x = xcenters .+ offset
        barplot!(ax, x, vals; width=bar_w*0.95, color=col, strokewidth=1.0, strokecolor=:black, label=name)

        # labels above bars (optional: comment out if crowded)
        @inbounds for (xi, yi) in zip(x, vals)
            text!(ax, xi, yi, text=@sprintf("%.4f", yi), align=(:center, :bottom), fontsize=10)
        end
    end

    ax.xticks = (xcenters, [@sprintf("%.1f", p) for p in p_grid])
    axislegend(ax; position=:rt, framevisible=true)

    save(out_png, fig)
    save(out_pdf, fig)

    println("==============================================================")
    println("Saved → $out_png")
    println("Saved → $out_pdf")
    println("RAW constants repeated across p: eq+spa=$(v_eqspa_const), jsdc=$(v_jsdc_const)")
    println("==============================================================")
end

main()
