module IEEEPlotBER
using PyPlot
const plt = PyPlot

export ieee_style!, pchip_log, plot_ber_snr

function ieee_style!(; figsize=(3.5, 2.6), font=9, label=11, tick=10, legend=9, lw=1.1)
    rc = PyPlot.PyDict(PyPlot.matplotlib["rcParams"])
    rc["figure.figsize"]  = figsize
    rc["font.size"]       = font
    rc["axes.labelsize"]  = label
    rc["xtick.labelsize"] = tick
    rc["ytick.labelsize"] = tick
    rc["legend.fontsize"] = legend
    rc["lines.linewidth"] = lw

    plt.tick_params(which="both", direction="in")
    plt.tick_params(axis="both", which="major", length=4, width=0.9)
    plt.tick_params(axis="both", which="minor", length=2, width=0.7)
    return nothing
end

function pchip_log(x::Vector{Float64}, y::Vector{Float64};
                   ngrid::Int=40, eps::Float64=1e-12, monotone::Bool=true)
    @assert length(x) == length(y) && length(x) >= 2
    p  = sortperm(x)
    xs = x[p]
    ys = clamp.(y[p], eps, Inf)
    ly = log10.(ys)

    n  = length(xs)
    dx = [xs[i+1]-xs[i] for i in 1:n-1]
    d  = [(ly[i+1]-ly[i])/dx[i] for i in 1:n-1]

    m = zeros(Float64, n)
    m[1] = d[1]
    m[end] = d[end]
    for i in 2:n-1
        if d[i-1]*d[i] <= 0
            m[i] = 0.0
        else
            w1 = 2*dx[i]   + dx[i-1]
            w2 = 2*dx[i-1] + dx[i]
            m[i] = (w1 + w2) / (w1/d[i-1] + w2/d[i])
        end
    end

    xg = collect(range(xs[1], xs[end], length=ngrid))
    yg = similar(xg)

    j = 1
    for t in eachindex(xg)
        xq = xg[t]
        while j < n && xq > xs[j+1]
            j += 1
        end
        j == n && (j = n-1)

        h  = xs[j+1]-xs[j]
        τ  = (xq - xs[j]) / h
        h00 = (1 + 2τ) * (1 - τ)^2
        h10 = τ * (1 - τ)^2
        h01 = τ^2 * (3 - 2τ)
        h11 = τ^2 * (τ - 1)

        lyq = h00*ly[j] + h*h10*m[j] + h01*ly[j+1] + h*h11*m[j+1]
        yg[t] = 10.0^lyq
    end

    if monotone
        for i in 2:length(yg)
            yg[i] = min(yg[i-1], yg[i])
        end
    end

    return xg, yg
end

function plot_ber_snr(snr::AbstractVector{<:Real},
                      bers::Vector{<:AbstractVector{<:Real}};
                      labels::Vector{String}=String[],
                      styles::Vector=Any[],
                      smooth::Bool=true,
                      grid_pts::Int=60,
                      enforce_mon::Bool=true,
                      title::String="BER vs SNR",
                      xlabel::String="SNR (dB)",
                      ylabel::String="BER",
                      legend_loc::String="upper right",
                      xlim=nothing,
                      ylim=nothing,
                      save_prefix::Union{Nothing,String}=nothing,
                      dpi::Int=300)

    ieee_style!()
    x = Float64.(collect(snr))

    ncurves = length(bers)
    @assert ncurves >= 1
    for i in 1:ncurves
        @assert length(bers[i]) == length(x) "bers[$i] length != snr length"
    end

    if isempty(labels)
        labels = ["curve$(i)" for i in 1:ncurves]
    else
        @assert length(labels) == ncurves
    end

    default_styles = [
        (color="black", ls="-",  marker="o", mfc="white", mec="black", mew=0.9),
        (color="0.55",  ls="--", marker="x", mfc="none",  mec="0.25", mew=1.0),
        (color="0.25",  ls="-.", marker="s", mfc="white", mec="0.25", mew=0.9),
        (color="0.75",  ls=":",  marker="^", mfc="white", mec="0.50", mew=0.9),
    ]

    plt.figure()
    all_raw_y = Float64[]

    for i in 1:ncurves
        yraw = Float64.(collect(bers[i]))
        append!(all_raw_y, yraw)

        xs, ys = x, yraw
        if smooth && length(xs) >= 2
            xg, yg = pchip_log(xs, ys; ngrid=grid_pts, monotone=enforce_mon)
        else
            xg, yg = xs, ys
        end

        st = default_styles[1 + (i-1) % length(default_styles)]
        if i <= length(styles) && styles[i] !== nothing
            st = merge(st, styles[i])
        end

        lw  = haskey(st, :lw)  ? st.lw  : 1.1
        ms  = haskey(st, :ms)  ? st.ms  : 4.5
        mfc = haskey(st, :mfc) ? st.mfc : "white"
        mec = haskey(st, :mec) ? st.mec : st.color
        mew = haskey(st, :mew) ? st.mew : 0.9
        markevery = max(1, Int(floor(length(xg)/8)))

        plt.semilogy(
            xg, yg;
            color=st.color, ls=st.ls, lw=lw,
            marker=st.marker, markersize=ms, markevery=markevery,
            markerfacecolor=mfc, markeredgecolor=mec, markeredgewidth=mew,
            label=labels[i]
        )
    end

    if xlim === nothing
        plt.xlim(minimum(x), maximum(x))
    else
        plt.xlim(xlim[1], xlim[2])
    end

    if ylim === nothing
        ymin = max(minimum(all_raw_y), 1e-12)
        ymax = maximum(all_raw_y)
        ylo  = 10.0^(floor(log10(ymin)))
        yhi  = 10.0^(ceil(log10(ymax)))
        plt.ylim(ylo, yhi)
    else
        plt.ylim(ylim[1], ylim[2])
    end

    plt.grid(which="both", alpha=0.28, ls=":", color="0.7")
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=legend_loc, frameon=false)
    plt.tight_layout()

    if save_prefix !== nothing
        plt.savefig(save_prefix * ".png", dpi=dpi)
        plt.savefig(save_prefix * ".pdf")
        println("Saved → ", save_prefix, ".png & .pdf")
    end

    return plt.gcf(), plt.gca()
end

end # module
