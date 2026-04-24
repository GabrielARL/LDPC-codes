#!/usr/bin/env julia
# scripts/plot_rsc_psrbar.jl
#
# VSCode-friendly grouped bar plot: RSC PSR vs pilot ratio p
#
# DEFAULT INPUT (prefers 4ways):
#   1) data/runs/psweep_replayswap_bpsk_RSC64_128_4ways.csv
#   2) data/runs/psweep_replayswap_bpsk_RSC64_128_3ways.csv
#   3) data/runs/compare_replayswap_bpsk_RSC64_128_TURBO.csv
#
# Usage:
#   julia --project=. scripts/plot_rsc_psrbar.jl
#   julia --project=. scripts/plot_rsc_psrbar.jl data/runs/psweep_replayswap_bpsk_RSC64_128_4ways.csv
#
# Options:
#   --out    <png path>
#   --outpdf <pdf path>
#   --series "eq,jsdc,turbo"       (default: auto from columns)
#   --metric u64|b128|b128_hard    (default u64)
#   --agg mean|median              (default mean)
#   --ps "0,0.1,0.2"               (filter/order)
#
# ----------------------------
# PATCH (2): "Am I plotting the right CSV?"
# ----------------------------
# This script now:
#   - prints the CSV path it is using
#   - prints ALL columns
#   - detects format:
#       (A) per-(p,blk) psweep format (your new linear/turbo debug psweep)
#           -> expects columns like :method, :u64_psr, :b128_post_psr
#       (B) legacy 3ways/4ways format
#           -> expects columns like :turbo_psr_u64, :jsdc_psr_u64, :eq_psr_u64, etc
#   - prints a clear "Detected format: ..." line so you KNOW.
#
using CSV, DataFrames, Statistics, Printf
using CairoMakie

# ----------------------------
# helpers
# ----------------------------
parse_floats(s::String) = isempty(strip(s)) ? Float64[] : Float64.(parse.(Float64, split(replace(strip(s)," "=>""), ",")))

function parse_series(s::String)
    t = replace(strip(s), " " => "")
    isempty(t) && return String[]
    return split(t, ",")
end

die(msg::String) = error(msg)

function pick_default_csv()
    cand = [
        "data/runs/psweep_replayswap_bpsk_RSC64_128_4ways.csv",
        "data/runs/psweep_replayswap_bpsk_RSC64_128_3ways.csv",
        "data/runs/compare_replayswap_bpsk_RSC64_128_TURBO.csv",
    ]
    for c in cand
        isfile(c) && return c
    end
    return cand[1]  # will error nicely later
end

function aggfun(name::String)
    name == "mean" && return mean
    name == "median" && return median
    die("--agg must be mean or median")
end

function first_existing_col(df::DataFrame, cands::Vector{Symbol})
    for c in cands
        hasproperty(df, c) && return c
    end
    return nothing
end

# --- CSV format detection (PATCH 2) ---
# A: your new per-(p,blk) psweep
function is_psweep_perblk_format(df::DataFrame)
    hasproperty(df, :p) &&
    hasproperty(df, :blk) &&
    hasproperty(df, :method) &&
    (hasproperty(df, :u64_psr) || hasproperty(df, :b128_post_psr) || hasproperty(df, :b128_post_ber))
end

# B: legacy 3ways/4ways (turbo/jsdc/eq columns)
function is_legacy_ways_format(df::DataFrame)
    hasproperty(df, :p) &&
    (hasproperty(df, :turbo_psr64) ||
     hasproperty(df, :turbo_psr_u64) ||
     hasproperty(df, :eq_psr64) ||
     hasproperty(df, :jsdc_psr64) ||
     hasproperty(df, :u64_psr))  # sometimes your compare CSV uses u64_psr
end

function describe_format(df::DataFrame, in_csv::String)
    if is_psweep_perblk_format(df)
        println("Detected format: per-(p,blk) psweep (method/u64_psr/b128_post_psr present). ✅")
        println("Tip: plot this CSV by aggregating u64_psr or b128_post_psr over blocks for each p.")
    elseif is_legacy_ways_format(df)
        println("Detected format: legacy 3ways/4ways (eq/jsdc/turbo columns). ✅")
        println("Tip: plot this CSV using --series eq,jsdc,turbo (default auto).")
    else
        println("Detected format: UNKNOWN. ❌")
        println("This CSV does not look like either expected format.")
        println("Check the header row / columns above.")
    end

    # Extra helpful hint if you accidentally used default-pick
    if occursin("psweep_replayswap_bpsk_RSC64_128_4ways", in_csv) &&
       is_psweep_perblk_format(df)
        println("WARNING: filename suggests 4ways but columns look like per-blk psweep. Double-check you passed the right file.")
    end
end

function series_metric_to_cols(series::String, metric::String)
    s = lowercase(series)
    m = lowercase(metric)

    if m == "u64"
        if s == "eq";        return [:eq_psr_u64, :u64_psr]
        elseif s == "jsdc";  return [:jsdc_psr_u64]
        elseif s == "turbo"; return [:turbo_psr_u64, :u64_psr]
        else die("Unknown series '$series'"); end

    elseif m == "b128"
        if s == "eq";        return [:eq_psr_b128, :b128_post_psr]
        elseif s == "jsdc";  return [:jsdc_psr_b128]
        elseif s == "turbo"; return [:turbo_psr_b128, :b128_post_psr]
        else die("Unknown series '$series'"); end

    elseif m == "b128_hard"
        s == "eq" || die("--metric b128_hard supports only --series eq")
        return [:eqhard_psr_b128]

    else
        die("--metric must be u64, b128, or b128_hard")
    end
end

# if user didn't specify --series, pick a default based on available columns
function default_series_for(df::DataFrame, metric::String)
    m = lowercase(metric)
    s = String[]
    if first_existing_col(df, series_metric_to_cols("eq", m)) !== nothing
        push!(s, "eq")
    end
    if first_existing_col(df, series_metric_to_cols("jsdc", m)) !== nothing
        push!(s, "jsdc")
    end
    if first_existing_col(df, series_metric_to_cols("turbo", m)) !== nothing
        push!(s, "turbo")
    end
    isempty(s) && die("Could not find any columns for metric='$metric' in CSV. Columns: $(names(df))")
    return s
end

# ----------------------------
# main
# ----------------------------
function main()
    # positional CSV (optional)
    in_csv = (length(ARGS) >= 1 && !startswith(ARGS[1], "--")) ? ARGS[1] : pick_default_csv()

    out_png = "data/plots/rsc_psrbar.png"
    out_pdf = "data/plots/rsc_psrbar.pdf"

    metric = "b128"     # u64|b128|b128_hard
    agg = "mean"
    ps_filter = Float64[]
    series = String[]  # empty means "auto"

    # parse flags
    i = (length(ARGS) >= 1 && !startswith(ARGS[1], "--")) ? 2 : 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--out"; i += 1; out_png = ARGS[i]
        elseif a == "--outpdf"; i += 1; out_pdf = ARGS[i]
        elseif a == "--series"; i += 1; series = parse_series(ARGS[i])
        elseif a == "--metric"; i += 1; metric = ARGS[i]
        elseif a == "--agg"; i += 1; agg = ARGS[i]
        elseif a == "--ps"; i += 1; ps_filter = parse_floats(ARGS[i])
        elseif a == "--help" || a == "-h"
            println("""
Usage:
  julia --project=. scripts/plot_rsc_psrbar.jl [csv] [args]

Args:
  --out    data/plots/rsc_psrbar.png
  --outpdf data/plots/rsc_psrbar.pdf
  --metric u64|b128|b128_hard     (default b128)
  --series "eq,jsdc,turbo"        (default: auto from columns)
  --agg    mean|median            (default mean)
  --ps     "0,0.1,0.2,0.3"        (filter/order)

Examples:
  julia --project=. scripts/plot_rsc_psrbar.jl --metric b128
  julia --project=. scripts/plot_rsc_psrbar.jl data/runs/psweep_replayswap_bpsk_RSC64_128_4ways.csv --metric u64 --series "eq,jsdc,turbo"
""")
            return
        else
            die("Unknown arg: $a")
        end
        i += 1
    end

    isfile(in_csv) || die("Missing CSV: $in_csv")

    # ----------------------------
    # READ + PRINT WHAT WE’RE USING (PATCH 2)
    # ----------------------------
    df = CSV.read(in_csv, DataFrame)
    println("==============================================================")
    println("Using CSV: ", in_csv)
    println("Rows: ", nrow(df), " | Cols: ", ncol(df))
    println("Columns:")
    println(names(df))
    describe_format(df, in_csv)
    println("==============================================================")

    hasproperty(df, :p) || die("CSV missing column :p")
    df.p = Float64.(df.p)

    # filter/order p
    if !isempty(ps_filter)
        want = Set(round.(ps_filter; digits=6))
        df = df[in.(round.(df.p; digits=6), Ref(want)), :]
    end
    nrow(df) > 0 || die("No rows after filtering.")

    ps = sort(unique(df.p))
    if !isempty(ps_filter)
        ps = ps_filter
    end

    if length(unique(df.p)) == 1
        @warn "CSV contains only one unique p value. You will get one group." in_csv unique_p=unique(df.p)
    end

    # auto series if not specified
    if isempty(series)
        series = default_series_for(df, metric)
    end
    series = [lowercase(s) for s in series]
    nS = length(series)
    nS > 0 || die("Empty --series")

    fagg = aggfun(lowercase(agg))

    # compute per-series aggregated values
    vals = Dict{String, Vector{Float64}}()
    for s in series
        col = first_existing_col(df, series_metric_to_cols(s, metric))
        col === nothing && die("Missing required column for series='$s' metric='$metric'. Columns: $(names(df))")
        df[!, col] = Float64.(df[!, col])

        v = Float64[]
        for p in ps
            sub = df[df.p .== p, col]
            isempty(sub) ? push!(v, NaN) : push!(v, fagg(sub))
        end
        vals[s] = v
    end

    # ----------------------------
    # plot grouped bars
    # ----------------------------
    mkpath(dirname(out_png))
    mkpath(dirname(out_pdf))

    title_str = lowercase(metric) == "u64" ? "RSC u64 PSR vs Pilot Ratio" :
                lowercase(metric) == "b128" ? "RSC b128 PSR vs Pilot Ratio" :
                "RSC b128 (EQ-hard) PSR vs Pilot Ratio"

    fig = Figure(size=(980, 480))
    ax = Axis(fig[1, 1],
        title = title_str,
        xlabel = "Pilot ratio p",
        ylabel = "PSR"
    )
    ylims!(ax, 0.0, 1.0)

    xcenters = collect(1:length(ps))
    group_w = 0.80
    bar_w = group_w / nS

    colors = Dict(
        "eq" => (:dodgerblue, 0.85),
        "jsdc" => (:orange, 0.85),
        "turbo" => (:seagreen, 0.85)
    )
    labels = Dict(
        "eq" => "EQ",
        "jsdc" => "JSDC",
        "turbo" => "TurboEQ"
    )

    for (k, s) in enumerate(series)
        offset = -group_w/2 + (k - 0.5) * bar_w
        x = xcenters .+ offset
        y = vals[s]

        barplot!(ax, x, y;
                 width=bar_w*0.95,
                 color=get(colors, s, (:gray, 0.85)),
                 strokewidth=1.0,
                 strokecolor=:black,
                 label=get(labels, s, s))

        @inbounds for (xi, yi) in zip(x, y)
            isfinite(yi) || continue
            text!(ax, xi, min(yi + 0.03, 0.98),
                  text=@sprintf("%.3f", yi),
                  align=(:center, :bottom),
                  fontsize=11)
        end
    end

    ax.xticks = (xcenters, [@sprintf("%.1f", p) for p in ps])
    ax.ygridvisible = true
    ax.xgridvisible = false
    axislegend(ax; position=:rb, framevisible=true)

    save(out_png, fig)
    save(out_pdf, fig)

    println("Saved → $out_png")
    println("Saved → $out_pdf")
end

main()
