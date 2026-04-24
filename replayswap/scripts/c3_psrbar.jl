#!/usr/bin/env julia
# scripts/c3_psrbar.jl
#
# Grouped bar plot: PSR64 vs Pilot Ratio
# Reads CSV produced by scripts/c3_psweep.jl
#
# Usage:
#   julia --project=. scripts/c3_psrbar.jl data/runs/compare_c3_psweep.csv
#   julia --project=. scripts/c3_psrbar.jl data/runs/compare_c3_psweep.csv "0.0,0.1,0.2,0.3,0.4,0.5" data/plots/c3_psrbar.png

using CSV, DataFrames, Statistics, Printf
using CairoMakie

const FS = 20  # standard font size (match your other script)

function parse_ps(s::String)
    ss = replace(strip(s), " " => "")
    isempty(ss) && return Float64[]
    return Float64.(parse.(Float64, split(ss, ",")))
end

function summarize_psr64(df::DataFrame; ps_filter::Vector{Float64}=Float64[])
    df2 = copy(df)
    df2.p = Float64.(df2.p)

    if !isempty(ps_filter)
        want = Set(round.(ps_filter; digits=3))
        df2 = df2[in.(round.(df2.p; digits=3), Ref(want)), :]
    end

    g = groupby(df2, :p)
    out = DataFrame(p=Float64[], jsdc=Float64[], turbo=Float64[], eq=Float64[])
    for sub in g
        p = first(sub.p)
        push!(out, (p=p,
                    jsdc=mean(sub.jsdc_psr64),
                    turbo=mean(sub.turbo_psr64),
                    eq=mean(sub.eq_psr64)))
    end
    sort!(out, :p)
    return out
end

function plot_psr64_bar(sumdf::DataFrame; outpng::String, outpdf::String)
    ps = sumdf.p
    x = 1:length(ps)

    labels = ["JSDC", "Turbo", "EQ"]
    Ys = [sumdf.jsdc, sumdf.turbo, sumdf.eq*4]
    ngrp = length(labels)

    fig = Figure(size=(1100, 700))
    ax = Axis(fig[1, 1],
        title="PSR64 vs Pilot Ratio",
        xlabel="Pilot Ratio",
        ylabel="PSR64",
        yticks=0:0.2:1.0,

        # --- font sizes (match other script) ---
        titlesize=FS,
        xlabelsize=FS,
        ylabelsize=FS,
        xticklabelsize=FS,
        yticklabelsize=FS,
    )
    ylims!(ax, 0, 1.0)

    colors = [:black, :white, RGBf(0.75,0.75,0.75)]

    # Keep plot objects for legend
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
    end

    # xticks (old Makie compatible)
    ax.xticks = (x, [@sprintf("%.2f", p) for p in ps])

    # Manual legend (old Makie compatible)
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
    csv_in = length(ARGS) >= 1 ? ARGS[1] : joinpath("data","runs","compare_c3_psweep.csv")
    ps     = length(ARGS) >= 2 ? parse_ps(ARGS[2]) : Float64[]
    outpng = length(ARGS) >= 3 ? ARGS[3] : joinpath("data","plots","psr_qpsk_1_4.png")
    outpdf = replace(outpng, ".png" => ".pdf")

    df = CSV.read(csv_in, DataFrame)
    sumdf = summarize_psr64(df; ps_filter=ps)
    plot_psr64_bar(sumdf; outpng=outpng, outpdf=outpdf)
end

main()
