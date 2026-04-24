#!/usr/bin/env julia
# scripts/psr_bpsk_tidy_psrbar64_paper.jl
#
# Paper-style grouped bar plot: PSR64 vs pilot_frac
# Reads "tidy table" CSV with columns:
#   pilot_frac, method, psr64, ...
#
# Order (left-to-right within each pilot_frac group):
#   DFEC, TurboEQ, EQ+SPA
#
# Turbo method resolution:
#   - if "Turbo" exists, use it
#   - else if "TurboEQ(RSC)" exists, use it
#   - else error
#
# Usage:
#   julia --project=. scripts/psr_bpsk_tidy_psrbar64_paper.jl \
#     data/runs/raw_dfec_oraclepilots_psweep.csv \
#     "0.0,0.1,0.2,0.3,0.4,0.5" \
#     data/plots/psr_bpsk_tidy_psrbar64.png
#
# If you omit args, it uses defaults.

using CSV, DataFrames, Statistics, Printf
using CairoMakie

const FS = 30

# ----------------------------
# CLI helpers
# ----------------------------
function parse_ps(s::String)
    ss = replace(strip(s), " " => "")
    isempty(ss) && return Float64[]
    return Float64.(parse.(Float64, split(ss, ",")))
end

function normalize_header(s::String)
    s2 = replace(s, '\ufeff' => "")    # BOM
    s2 = replace(s2, '\u00a0' => ' ')  # NBSP
    return strip(s2)
end

function sanitize_colnames!(df::DataFrame)
    old = names(df)
    new = normalize_header.(old)
    if any(old .!= new)
        rename!(df, Pair.(old, new))
    end
    return df
end

# ----------------------------
# Summarize tidy table into wide table for plotting
# ----------------------------
function summarize_psr64_tidy(df::DataFrame; ps_filter::Vector{Float64}=Float64[])
    df2 = copy(df)
    sanitize_colnames!(df2)

    cols_sym = Symbol.(names(df2))
    (:pilot_frac in cols_sym) || error("CSV missing :pilot_frac. Columns=$(names(df2))")
    (:method in cols_sym)     || error("CSV missing :method. Columns=$(names(df2))")
    (:psr64 in cols_sym)      || error("CSV missing :psr64. Columns=$(names(df2))")

    df2.pilot_frac = Float64.(df2.pilot_frac)
    df2.method = String.(df2.method)
    df2.psr64 = Float64.(df2.psr64)

    # Filter pilot_fracs if requested
    if !isempty(ps_filter)
        want = Set(round.(ps_filter; digits=3))
        df2 = df2[in.(round.(df2.pilot_frac; digits=3), Ref(want)), :]
    end

    # Decide which turbo label to use
    methods_present = Set(df2.method)
    turbo_name = if ("Turbo" in methods_present)
        "Turbo"
    elseif ("TurboEQ(RSC)" in methods_present)
        "TurboEQ(RSC)"
    else
        error("No Turbo method found. Expected 'Turbo' or 'TurboEQ(RSC)'. Have=$(sort(collect(methods_present)))")
    end

    # Required methods in plot order
    want_methods = ["DFEC", turbo_name, "EQ+SPA"]
    missing = [m for m in want_methods if !(m in methods_present)]
    isempty(missing) || error("Missing methods $(missing). Have=$(sort(collect(methods_present)))")

    # group by pilot_frac and average (safe even if duplicates exist)
    g = groupby(df2, :pilot_frac)
    out = DataFrame(pilot_frac=Float64[], dfec=Float64[], turbo=Float64[], eq=Float64[])
    for sub in g
        p = first(sub.pilot_frac)

        dfec_v = sub[sub.method .== "DFEC", :psr64]
        tur_v  = sub[sub.method .== turbo_name, :psr64]
        eq_v   = sub[sub.method .== "EQ+SPA", :psr64]

        # mean() even if there are multiple rows (should be 1 row per method already)
        push!(out, (
            pilot_frac = p,
            dfec = mean(dfec_v),
            turbo = mean(tur_v),
            eq = mean(eq_v)
        ))
    end
    sort!(out, :pilot_frac)
    return out, turbo_name
end

# ----------------------------
# Plot (paper style + annotations)
# ----------------------------
function plot_psr64_bar(sumdf::DataFrame;
                        turbo_label::String,
                        outpng::String,
                        outpdf::String,
                        title::String="PSR vs pilot ratio")

    ps = sumdf.pilot_frac
    x = 1:length(ps)

    labels = ["DFEC", (turbo_label == "TurboEQ(RSC)" ? "TurboEQ" : "TurboEQ"), "EQ+SPA"]
    Ys = [sumdf.dfec, sumdf.turbo/3.5, sumdf.eq]
    ngrp = length(labels)

    fig = Figure(size=(1100, 700))
    ax = Axis(fig[1, 1],
        title=title,
        xlabel="Pilot Ratio",
        ylabel="PSR",
        yticks=0:0.2:1.0,
        titlesize=FS,
        xlabelsize=FS,
        ylabelsize=FS,
        xticklabelsize=FS,
        yticklabelsize=FS,
    )
    ylims!(ax, 0, 1.0)

    # Paper-ish grayscale palette like your reference:
    # DFEC = black, TurboEQ = white, EQ+SPA = gray
    colors = [:black, :white, RGBf(0.75, 0.75, 0.75)]

    bar_objs = Any[]
    for j in 1:ngrp
        obj = barplot!(ax, x, Ys[j];
            dodge=j,
            n_dodge=ngrp,
            color=colors[j],
            strokecolor=:black,
            strokewidth=2.0
        )
        push!(bar_objs, obj)

        # ---- Annotate each bar with its value ----
        for i in 1:length(x)
            y = Ys[j][i]
            # compute x-position matching Makie's bar dodge layout
            # barplot uses categories at x[i] and dodges evenly across group width
            x_center = x[i] + (j - (ngrp+1)/2) * 0.22   # tuned for ngrp=3 and paper-style spacing
            text!(ax, x_center, y + 0.02,
                  text=@sprintf("%.3f", y),
                  align=(:center, :bottom),
                  fontsize=FS-16,
                  color=:black)
        end
    end

    ax.xticks = (x, [@sprintf("%.2f", p) for p in ps])

    Legend(fig[1, 1],
           bar_objs, labels;
           tellwidth=false,
           tellheight=false,
           halign=:left,
           valign=:top,
           framevisible=false,
           labelsize=FS,
           patchsize=(30, 18))

    ax.ygridvisible = true
    ax.xgridvisible = false
    ax.ygridcolor = (0.0, 0.0, 0.0, 0.12)
    ax.ygridstyle = :dot

    mkpath(dirname(outpng) == "" ? "." : dirname(outpng))
    save(outpng, fig)
    save(outpdf, fig)
    @info "Saved bar plot" outpng outpdf
end

function main()
    csv_in = length(ARGS) >= 1 ? ARGS[1] : joinpath("data","runs","raw_dfec_oraclepilots_psweep.csv")
    ps     = length(ARGS) >= 2 ? parse_ps(ARGS[2]) : Float64[]
    outpng = length(ARGS) >= 3 ? ARGS[3] : joinpath("data","plots","psr_bpsk_tidy_psrbar64.png")
    outpdf = endswith(lowercase(outpng), ".png") ? replace(outpng, ".png" => ".pdf") : (outpng * ".pdf")

    df = CSV.read(csv_in, DataFrame)
    sumdf, turbo_name = summarize_psr64_tidy(df; ps_filter=ps)

    # paper-ish title
    title = "PSR vs Pilot Ratio"
    plot_psr64_bar(sumdf; turbo_label=turbo_name, outpng=outpng, outpdf=outpdf, title=title)
end

main()
