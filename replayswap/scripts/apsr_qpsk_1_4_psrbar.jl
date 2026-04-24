#!/usr/bin/env julia
# scripts/psr_qpsk_1_4_psrbar.jl
#
# Grouped bar plot: PSR vs Pilot Ratio
# Reads CSV produced by scripts/psr_qpsk_1_4_psweep.jl
#
# Usage:
#   julia --project=. scripts/psr_qpsk_1_4_psrbar.jl
#   julia --project=. scripts/psr_qpsk_1_4_psrbar.jl data/runs/psr_qpsk_1_4_psweep.csv
#   julia --project=. scripts/psr_qpsk_1_4_psrbar.jl data/runs/psr_qpsk_1_4_psweep.csv "0.0,0.1,0.2,0.3,0.4,0.5" data/plots/psr_qpsk_1_4_psrbar.png

using CSV, DataFrames, Statistics, Printf
using CairoMakie

const FS = 30

function parse_ps(s::String)
    ss = replace(strip(s), " " => "")
    isempty(ss) && return Float64[]
    return Float64.(parse.(Float64, split(ss, ",")))
end

function pick_col(cols_sym::Vector{Symbol}, cands::Vector{Symbol}, who::String; allcols=nothing)
    for c in cands
        if c in cols_sym
            return c
        end
    end
    msg = "CSV missing $(who) column. Tried $(cands)."
    if allcols !== nothing
        msg *= " Columns=$(allcols)"
    end
    error(msg)
end

function summarize_psr(df::DataFrame; ps_filter::Vector{Float64}=Float64[])
    df2 = copy(df)
    cols_sym = Symbol.(names(df2))

    (:p in cols_sym) || error("CSV missing column :p. Columns=$(names(df2))")
    df2.p = Float64.(df2.p)

    # allow older column names too
    col_dfec = pick_col(cols_sym,
        [:dfec_psr, :dfec_psr64, :jsdc_psr, :jsdc_psr64],
        "DFEC PSR"; allcols=names(df2)
    )
    col_turbo = pick_col(cols_sym,
        [:turboeq_psr, :turboeq_psr64, :turbo_psr, :turbo_psr64],
        "Turbo EQ PSR"; allcols=names(df2)
    )
    col_eqspa = pick_col(cols_sym,
        [:eqspa_psr, :eqspa_psr64, :eq_psr, :eq_psr64],
        "EQ+SPA PSR"; allcols=names(df2)
    )

    if !isempty(ps_filter)
        want = Set(round.(ps_filter; digits=3))
        df2 = df2[in.(round.(df2.p; digits=3), Ref(want)), :]
    end

    g = groupby(df2, :p)
    out = DataFrame(p=Float64[], dfec=Float64[], turboeq=Float64[], eqspa=Float64[])
    for sub in g
        p = first(sub.p)
        push!(out, (p=p,
                    dfec=mean(Float64.(sub[!, col_dfec])),
                    turboeq=mean(Float64.(sub[!, col_turbo])),
                    eqspa=mean(Float64.(sub[!, col_eqspa]))))
    end
    sort!(out, :p)
    return out
end

# Compute the x-center of the j-th dodged bar in the i-th group,
# matching Makie's (width, dodge_gap, n_dodge) geometry.
@inline function dodged_xcenter(xi::Real, j::Int, ngrp::Int; width::Float64=0.80, dodge_gap::Float64=0.06)
    # total span = width
    # inside: n bars with (ngrp-1) gaps of dodge_gap
    barw = (width - dodge_gap*(ngrp-1)) / ngrp
    left = xi - width/2
    return left + (j-1)*(barw + dodge_gap) + barw/2
end

function plot_psr_bar(sumdf::DataFrame; outpng::String, outpdf::String, title::String="PSR Vs Pilot Ratio")
    ps = sumdf.p
    x  = 1:length(ps)

    labels = ["DFEC", "Turbo EQ", "EQ+SPA"]
    Ys     = [sumdf.dfec, sumdf.turboeq, sumdf.eqspa]
    ngrp   = length(labels)

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

    # headroom so labels above bars never clip
    ylims!(ax, 0, 1.06)

    colors = [:black, :white, RGBf(0.75, 0.75, 0.75)]

    width = 0.80
    dodge_gap = 0.06

    bar_objs = Any[]
    for j in 1:ngrp
        vals = Ys[j]

        obj = barplot!(ax, x, vals;
            dodge=j,
            n_dodge=ngrp,
            width=width,
            dodge_gap=dodge_gap,
            color=colors[j],
            strokecolor=:black,
            strokewidth=2.0
        )
        push!(bar_objs, obj)

        # --- annotate ON TOP (paper-script style), using matched dodged centers ---
        for i in 1:length(x)
            yv = vals[i]
            xv = dodged_xcenter(x[i], j, ngrp; width=width, dodge_gap=dodge_gap)

            # small vertical offset; slightly larger for tiny bars so it stays readable
            dy = (yv < 0.08) ? 0.02 : 0.015
            text!(ax, xv, yv + dy,
                  text=@sprintf("%.3f", yv),
                  align=(:center, :bottom),
                  fontsize=FS - 14,
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
    csv_in = length(ARGS) >= 1 ? ARGS[1] : joinpath("data","runs","psr_qpsk_1_4_psweep.csv")
    ps     = length(ARGS) >= 2 ? parse_ps(ARGS[2]) : Float64[]
    outpng = length(ARGS) >= 3 ? ARGS[3] : joinpath("data","plots","psr_qpsk_1_4_psrbar.png")
    outpdf = endswith(lowercase(outpng), ".png") ? replace(outpng, ".png" => ".pdf") : (outpng * ".pdf")

    df = CSV.read(csv_in, DataFrame)
    sumdf = summarize_psr(df; ps_filter=ps)

    plot_psr_bar(sumdf; outpng=outpng, outpdf=outpdf, title="PSR Vs Pilot Ratio")
end

main()
