#!/usr/bin/env julia
# scripts/psr_bpsk_1_2_groupbar.jl
#
# DEFAULT behavior (no args):
#   - Plot RAW psr64 from: data/runs/psr_bpsk_1_2_raw_detail.csv
#   - Add ONE extra bar series from RSC: b128_post_psr from:
#       data/runs/psr_bpsk_1_2_rsc_detail.csv
#   - Default output:
#       data/plots/raw_psr64_plus_rsc.png  (+ .pdf)
#
# Only options you typically need:
#   --rsc_method TurboEQ|LinearEQ     (default TurboEQ)
#   --showjsdcspa 0|1                (default 1)
#   --showjsdc 0|1                   (default 0)   <-- NEW
#   --showeqspa 0|1                  (default 1)   <-- NEW
#
# Optional:
#   positional: [csv_in] [ps_list] [outpng]
#   --annotate 0|1
#
using CSV, DataFrames, Statistics, Printf
using CairoMakie

const FS = 22

# ----------------------------
# Helpers
# ----------------------------
function parse_ps(s::String)
    ss = replace(strip(s), " " => "")
    isempty(ss) && return Float64[]
    return Float64.(parse.(Float64, split(ss, ",")))
end

@inline function nanmean(v::AbstractVector{<:Real})
    good = v[.!isnan.(v)]
    isempty(good) ? NaN : mean(good)
end

function parse_args(args::Vector{String})
    positional = String[]

    # ---- DEFAULTS requested by you ----
    csv_in   = joinpath("data","runs","psr_bpsk_1_2_raw_detail.csv")
    outpng   = joinpath("data","plots","raw_psr64_plus_rsc.png")
    metric   = "psr64"
    annotate = true

    # RSC addon defaults
    add_rsc    = true
    rsc_csv    = joinpath("data","runs","psr_bpsk_1_2_rsc_detail.csv")
    rsc_method = "TurboEQ"                 # user can override
    rsc_metric = "b128_post_psr"           # fixed default
    rsc_label  = ""                        # auto from rsc_method

    # RAW method toggles
    showjsdcspa = true    # default YES
    showjsdc    = false   # default NO  <-- NEW
    showeqspa   = true    # default YES <-- NEW

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--rsc_method"
            i == length(args) && error("--rsc_method needs TurboEQ or LinearEQ")
            rsc_method = strip(args[i+1]); i += 2
        elseif a == "--showjsdcspa"
            i == length(args) && error("--showjsdcspa needs 0|1")
            showjsdcspa = (parse(Int, strip(args[i+1])) != 0); i += 2
        elseif a == "--showjsdc"
            i == length(args) && error("--showjsdc needs 0|1")
            showjsdc = (parse(Int, strip(args[i+1])) != 0); i += 2
        elseif a == "--showeqspa"
            i == length(args) && error("--showeqspa needs 0|1")
            showeqspa = (parse(Int, strip(args[i+1])) != 0); i += 2
        elseif a == "--annotate"
            i == length(args) && error("--annotate needs 0|1")
            annotate = (parse(Int, strip(args[i+1])) != 0); i += 2

        # optional overrides (kept available)
        elseif a == "--metric"
            i == length(args) && error("--metric needs a value")
            metric = strip(args[i+1]); i += 2
        elseif a == "--add_rsc"
            i == length(args) && error("--add_rsc needs 0|1")
            add_rsc = (parse(Int, strip(args[i+1])) != 0); i += 2
        elseif a == "--rsc_csv"
            i == length(args) && error("--rsc_csv needs a path")
            rsc_csv = args[i+1]; i += 2
        elseif a == "--rsc_metric"
            i == length(args) && error("--rsc_metric needs a column name")
            rsc_metric = strip(args[i+1]); i += 2
        elseif a == "--rsc_label"
            i == length(args) && error("--rsc_label needs a string")
            rsc_label = strip(args[i+1]); i += 2
        else
            push!(positional, a); i += 1
        end
    end

    # positional overrides (if user supplies them)
    if length(positional) >= 1 && !isempty(strip(positional[1]))
        csv_in = positional[1]
    end
    ps_str = length(positional) >= 2 ? positional[2] : ""
    if length(positional) >= 3 && !isempty(strip(positional[3]))
        outpng = positional[3]
    end
    ps = isempty(strip(ps_str)) ? Float64[] : parse_ps(ps_str)

    # auto label if not explicitly provided
    if isempty(rsc_label)
        rsc_label = "RSC $(rsc_method) b128_psr"
    end

    return (;
        csv_in, ps, outpng,
        metric, annotate,
        add_rsc, rsc_csv, rsc_method, rsc_metric, rsc_label,
        showjsdcspa, showjsdc, showeqspa
    )
end

function summarize_from_detail(df::DataFrame; metric::Symbol, ps_filter::Vector{Float64}=Float64[])
    (:p in Symbol.(names(df))) || error("Detail CSV missing column p. Columns=$(names(df))")
    (:method in Symbol.(names(df))) || error("Detail CSV missing column method. Columns=$(names(df))")
    (metric in Symbol.(names(df))) || error("Detail CSV missing metric $(metric). Columns=$(names(df))")

    df2 = copy(df)
    df2.p = Float64.(df2.p)
    df2.method = String.(df2.method)

    if !isempty(ps_filter)
        want = Set(round.(ps_filter; digits=6))
        df2 = df2[in.(round.(df2.p; digits=6), Ref(want)), :]
    end

    g = combine(groupby(df2, [:p, :method]), metric => nanmean => :metric_val)
    sort!(g, [:p, :method])
    return g
end

function plot_groupbar(g::DataFrame; outpng::String, outpdf::String, title::String, ylabel::String,
                       annotate::Bool, methods_order::Vector{String}=String[])
    ps = sort(unique(Float64.(g.p)))
    meths = unique(String.(g.method))

    methods = if !isempty(methods_order)
        [m for m in methods_order if m in meths]
    else
        sort(meths)
    end
    isempty(methods) && error("No methods to plot. Available=$(meths)")

    vals = Dict(m => [begin
        sub = g[(g.p .== p) .& (g.method .== m), :metric_val]
        isempty(sub) ? NaN : Float64(sub[1])
    end for p in ps] for m in methods)

    fig = Figure(size=(1600, 900))
    ax = Axis(fig[1,1],
        title=title,
        xlabel="Pilot Ratio p",
        ylabel=ylabel,
        yticks=0:0.2:1.0,
        titlesize=FS, xlabelsize=FS, ylabelsize=FS,
        xticklabelsize=FS, yticklabelsize=FS
    )
    ylims!(ax, 0, 1.0)

    ax.ygridvisible = true
    ax.xgridvisible = false
    ax.ygridcolor = (0,0,0,0.12)
    ax.ygridstyle = :dot

    xcenters = collect(1:length(ps))
    group_w = 0.90
    bar_w = group_w / length(methods)

    palette = [:white, RGBf(0.75,0.75,0.75), :black, :dodgerblue, :orange, :seagreen, :purple, :crimson]
    bar_objs = Any[]

    for (k, m) in enumerate(methods)
        offset = -group_w/2 + (k - 0.5)*bar_w
        x = xcenters .+ offset
        y = vals[m]
        col = palette[1 + mod(k-1, length(palette))]

        obj = barplot!(ax, x, y;
            width=bar_w*0.95,
            color=col,
            strokecolor=:black,
            strokewidth=2.0,
            label=m
        )
        push!(bar_objs, obj)

        if annotate
            for (xi, yi) in zip(x, y)
                isfinite(yi) || continue
                text!(ax, xi, min(1.0, yi + 0.02);
                      text=@sprintf("%.3f", yi),
                      align=(:center,:bottom),
                      fontsize=FS-4,
                      color=:black)
            end
        end
    end

    ax.xticks = (xcenters, [@sprintf("%.2f", p) for p in ps])

    Legend(fig[1,1], bar_objs, methods;
           tellwidth=false, tellheight=false,
           halign=:left, valign=:top,
           framevisible=false,
           labelsize=FS,
           patchsize=(30, 18))

    mkpath(dirname(outpng) == "" ? "." : dirname(outpng))
    save(outpng, fig); save(outpdf, fig)
    @info "Saved groupbar" outpng outpdf
end

# ----------------------------
# Main
# ----------------------------
function main()
    cfg = parse_args(ARGS)
    outpdf = endswith(lowercase(cfg.outpng), ".png") ? replace(cfg.outpng, ".png" => ".pdf") : (cfg.outpng * ".pdf")

    isfile(cfg.csv_in) || error("Missing csv_in: $(cfg.csv_in)")
    df = CSV.read(cfg.csv_in, DataFrame)

    # RAW method filtering
    if :method in Symbol.(names(df))
        df.method = String.(df.method)
        keep = trues(nrow(df))
        @inbounds for i in 1:nrow(df)
            m = df.method[i]
            if m == "EQ+SPA" && !cfg.showeqspa
                keep[i] = false
            elseif m == "JSDC" && !cfg.showjsdc
                keep[i] = false
            elseif m == "JSDC+SPA" && !cfg.showjsdcspa
                keep[i] = false
            end
        end
        df = df[keep, :]
    end

    metric_sym = Symbol(cfg.metric)
    g = summarize_from_detail(df; metric=metric_sym, ps_filter=cfg.ps)

    # Add one extra bar series from RSC detail (default ON)
    if cfg.add_rsc
        isfile(cfg.rsc_csv) || error("Missing rsc_csv: $(cfg.rsc_csv)")
        dfR = CSV.read(cfg.rsc_csv, DataFrame)
        (:method in Symbol.(names(dfR))) || error("RSC csv missing method column: $(cfg.rsc_csv)")
        dfR.method = String.(dfR.method)
        dfR = dfR[lowercase.(dfR.method) .== lowercase(cfg.rsc_method), :]
        nrow(dfR) > 0 || error("No rows for rsc_method=$(cfg.rsc_method) in $(cfg.rsc_csv)")

        rmetric = Symbol(cfg.rsc_metric)
        gR = summarize_from_detail(dfR; metric=rmetric, ps_filter=cfg.ps)
        gR.method = fill(cfg.rsc_label, nrow(gR))
        g = vcat(g, gR)
    end

    # Nice default legend order
    methods_order = String[]
    if :method in Symbol.(names(df))
        raw_methods = unique(String.(df.method))
        if cfg.showeqspa && ("EQ+SPA" in raw_methods); push!(methods_order, "EQ+SPA") end
        if cfg.showjsdc && ("JSDC" in raw_methods); push!(methods_order, "JSDC") end
        if cfg.showjsdcspa && ("JSDC+SPA" in raw_methods); push!(methods_order, "JSDC+SPA") end
    end
    if cfg.add_rsc
        push!(methods_order, cfg.rsc_label)
    end

    ttl = "RAW $(cfg.metric) + RSC $(cfg.rsc_method) $(cfg.rsc_metric) vs p (psr_bpsk_1_2)"
    ylab = cfg.metric

    plot_groupbar(g; outpng=cfg.outpng, outpdf=outpdf,
                  title=ttl, ylabel=ylab,
                  annotate=cfg.annotate,
                  methods_order=methods_order)
end

main()
