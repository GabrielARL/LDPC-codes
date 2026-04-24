# scripts/c3_grid_utils.jl
#
# Utilities + main() for constellation grid plotting.
# Intended to be included from scripts/c3_grid.jl as:
#   include(joinpath(@__DIR__, "c3_grid_utils.jl"))
#   using .C3GridUtils

module C3GridUtils

using JLD2
using CairoMakie
using Printf
using Glob
using Random
using Statistics
using CSV
using DataFrames

export main, parse_ps, parse_cols

const QPSK_SYM_MAP = ComplexF64[
    (-1 - 1im) / sqrt(2),
    (-1 + 1im) / sqrt(2),
    ( 1 - 1im) / sqrt(2),
    ( 1 + 1im) / sqrt(2),
]

# For bounded soft-symbol boxes from tanh(L/2):
# I,Q ∈ [-1/sqrt(2), +1/sqrt(2)]
const A_BOX = 1 / sqrt(2)

# -----------------------------
# Plot rotation helper
# -----------------------------
function rot_from_deg(deg::Int)::ComplexF64
    d = mod(deg, 360)
    if d == 0
        return 1 + 0im
    elseif d == 90
        return 0 + 1im
    elseif d == 180
        return -1 + 0im
    elseif d == 270
        return 0 - 1im
    else
        error("rot_deg must be one of 0,90,180,270 (got $deg)")
    end
end

function parse_ps(s::String)
    ss = replace(strip(s), " " => "")
    isempty(ss) && return Float64[]
    return Float64.(parse.(Float64, split(ss, ",")))
end

function list_jld2_inputs(input::String)
    if occursin("*", input) || occursin("?", input) || occursin("[", input)
        return sort(filter(isfile, glob(input)))
    end
    if isdir(input)
        return sort(filter(f -> endswith(f, ".jld2"), readdir(input; join=true)))
    end
    return String[]
end

function find_inputs(input::String)
    files = list_jld2_inputs(input)
    !isempty(files) && return files

    candidates = String[
        input,
        joinpath("data", "runs_comp3ways_constellations"),
        joinpath("data", "runs_comp3ways_constellations_v2"),
        joinpath("data", "runs_comp3ways_constellations", "*.jld2"),
        joinpath("data", "*.jld2"),
    ]
    for c in candidates
        files = list_jld2_inputs(c)
        if !isempty(files)
            println("[grid] input not found: \"$input\"")
            println("[grid] using: \"$c\"  (found $(length(files)) files)")
            return files
        end
    end
    error("No constellation .jld2 files found.")
end

function load_meta_p(path::String)
    d = JLD2.load(path)
    if haskey(d, "meta")
        meta = d["meta"]
        if meta isa NamedTuple && hasproperty(meta, :p)
            return Float64(getproperty(meta, :p))
        elseif meta isa AbstractDict
            if haskey(meta, :p); return Float64(meta[:p]) end
            if haskey(meta, "p"); return Float64(meta["p"]) end
        end
    end
    m = match(r"p([0-9]*\.?[0-9]+)", basename(path))
    m === nothing && return NaN
    return parse(Float64, m.captures[1])
end

# -----------------------------
# Frame filter (for bounded square)
# -----------------------------
function filter_square_frame(pts::Vector{ComplexF64}; edge_tol::Float64=0.02)
    edge_tol <= 0 && return pts
    isempty(pts) && return pts
    a = A_BOX
    keep = BitVector(undef, length(pts))
    @inbounds for i in eachindex(pts)
        I = abs(real(pts[i]))
        Q = abs(imag(pts[i]))
        on_edge = (abs(I - a) <= edge_tol) || (abs(Q - a) <= edge_tol)
        keep[i] = !on_edge
    end
    return pts[keep]
end

# -----------------------------
# EQ "unboxing": atanh map
# (-a,a) -> (-Inf,Inf) per dim
# -----------------------------
unbox1(x::Real; a::Real=A_BOX, eps::Real=1e-6) = atanh(clamp(x / a, -1 + eps, 1 - eps))

function unbox_pts(pts::Vector{ComplexF64}; a::Real=A_BOX, eps::Real=1e-6)
    out = Vector{ComplexF64}(undef, length(pts))
    @inbounds for i in eachindex(pts)
        out[i] = unbox1(real(pts[i]); a=a, eps=eps) + 1im * unbox1(imag(pts[i]); a=a, eps=eps)
    end
    return out
end

# For nicer viewing: pick symmetric limits from quantiles (ignore huge outliers)
function sym_lim_from_quantile(x::Vector{Float64}; q::Float64=0.995, minlim::Float64=1.2)
    isempty(x) && return (-minlim, minlim)
    xs = sort(abs.(x))
    k = clamp(Int(ceil(q * length(xs))), 1, length(xs))
    m = max(xs[k], minlim)
    return (-m, m)
end

# -----------------------------
# BER -> 4 Gaussian blobs (AWGN proxy)
# -----------------------------
function invnormcdf(p::Float64)
    p <= 0.0 && return -Inf
    p >= 1.0 && return Inf

    a = (-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00)
    d = ( 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00)

    plow  = 0.02425
    phigh = 1.0 - plow

    if p < plow
        q = sqrt(-2.0 * log(p))
        num = (((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6])
        den = ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1.0)
        return num / den
    elseif p > phigh
        q = sqrt(-2.0 * log(1.0 - p))
        num = -(((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6])
        den = ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1.0)
        return num / den
    else
        q = p - 0.5
        r = q*q
        num = (((((a[1]*r + a[2])*r + a[3])*r + a[4])*r + a[5])*r + a[6]) * q
        den = (((((b[1]*r + b[2])*r + b[3])*r + b[4])*r + b[5])*r + 1.0)
        return num / den
    end
end

qinv_ber(ber::Float64) = invnormcdf(1.0 - ber)

function blobs_from_ber(n::Int, ber::Float64; seed::Int=123)
    rng = MersenneTwister(seed)
    ber = clamp(ber, 1e-12, 0.5 - 1e-12)

    x = qinv_ber(ber)
    N0 = 1.0 / (x^2)           # Es=1 QPSK approx
    sigma = sqrt(N0 / 2.0)     # per real dimension

    pts = Vector{ComplexF64}(undef, n)
    @inbounds for k in 1:n
        c = QPSK_SYM_MAP[rand(rng, 1:4)]
        pts[k] = c + sigma*randn(rng) + 1im*sigma*randn(rng)
    end
    return pts
end

function load_ber_map(csv_path::String, col::Symbol)
    isfile(csv_path) || error("CSV not found: $csv_path")
    df = CSV.read(csv_path, DataFrame)

    ("p" in names(df)) || error("CSV missing column \"p\". Have: $(names(df))")

    colname = String(col)
    (colname in names(df)) || error("CSV missing column \"$colname\". Have: $(names(df))")

    p_round = round.(Float64.(df[!, "p"]); digits=3)
    ber_vals = Float64.(df[!, colname])

    tmp = DataFrame(p_round=p_round, ber=ber_vals)
    g = groupby(tmp, :p_round)

    m = Dict{Float64,Float64}()
    for sub in g
        p = first(sub.p_round)
        m[p] = mean(sub.ber)
    end
    return m
end

function map_to_nearest(requested::Vector{Float64}, available::Vector{Float64})
    mapped = Float64[]
    for pr in requested
        idx = argmin(abs.(available .- pr))
        push!(mapped, available[idx])
    end
    return mapped
end

function parse_cols(s::String)
    t = lowercase(replace(strip(s), " " => ""))
    isempty(t) && return Set{Symbol}()
    parts = split(t, ",")
    out = Set{Symbol}()
    for p in parts
        if p == "jsdc"; push!(out, :jsdc)
        elseif p == "turbo"; push!(out, :turbo)
        elseif p == "eq"; push!(out, :eq)
        else
            error("Unknown entry: $p (use jsdc,turbo,eq)")
        end
    end
    return out
end

function _panel!(fig, r::Int, c::Int, pts::Vector{ComplexF64}, ttl::String;
                 edge_tol::Float64,
                 rot_plot::ComplexF64,
                 use_ber_blob::Bool,
                 ber_val::Float64,
                 ber_n::Int,
                 ber_floor::Float64,
                 ber_ceil::Float64,
                 unbox::Bool=false,
                 unbox_eps::Float64=1e-6,
                 auto_unbox_lim::Bool=true,
                 unbox_q::Float64=0.995)


    ax = Axis(fig[r, c],
        title = ttl,
        xlabel = "I",
        ylabel = "Q",
        aspect = DataAspect(),
        xgridvisible = true,
        ygridvisible = true
    )

    pts2 = pts
    if use_ber_blob
        b = clamp(ber_val, ber_floor, ber_ceil)
        n = (ber_n > 0) ? ber_n : max(10_000, length(pts))
        pts2 = blobs_from_ber(n, b; seed=777 + 1000*r + 10*c)
    else
        # only do "box-edge shaving" on the bounded domain;
        # for unboxed plots it makes no sense.
        if !unbox
            pts2 = filter_square_frame(pts2; edge_tol=edge_tol)
        end
    end

    # rotate first
    pts2 = rot_plot .* pts2

    # reference points
    if unbox
        # nudge ref points inside the boundary so unbox() is finite
        ref_in = rot_plot .* ((1 - 10unbox_eps) .* QPSK_SYM_MAP)
        pts2 = unbox_pts(pts2; eps=unbox_eps)
        ref  = unbox_pts(ComplexF64.(ref_in); eps=unbox_eps)
    else
        ref  = rot_plot .* QPSK_SYM_MAP
    end

    scatter!(ax, real.(pts2), imag.(pts2); markersize=1.8, color=(:dodgerblue, 0.18))
    scatter!(ax, real.(ref),  imag.(ref);  markersize=10, color=(:tomato, 0.9))
    hlines!(ax, [0.0]; color=(:gray, 0.35))
    vlines!(ax, [0.0]; color=(:gray, 0.35))

    if unbox && auto_unbox_lim
        xl = sym_lim_from_quantile(real.(pts2); q=unbox_q, minlim=3.0)
        yl = sym_lim_from_quantile(imag.(pts2); q=unbox_q, minlim=3.0)
        xlims!(ax, xl...); ylims!(ax, yl...)
    else
        lim = 1.2
        xlims!(ax, -lim, lim); ylims!(ax, -lim, lim)
    end

    return ax
end

function main(; input::String,
              ps_req::Vector{Float64}=Float64[],
              out::String=joinpath("data","plots","c3_grid.png"),
              edge_tol::Float64=0.05,
              auto_nearest::Bool=true,
              rot_deg::Int=0,
              csv_ber::String=joinpath("data","runs","compare_c3_psweep.csv"),
              use_ber_blobs::Bool=true,
              ber_n::Int=80000,
              ber_floor::Float64=1e-4,
              ber_ceil::Float64=0.49,
              ber_cols::Set{Symbol}=Set([:jsdc, :turbo]),
              # Unboxing knobs (EQ)
              eq_unbox::Bool=true,
              unbox_eps::Float64=1e-6,
              unbox_q::Float64=0.995)

    files = find_inputs(input)
    println("[grid] scanning $(length(files)) files...")

    p2path = Dict{Float64,String}()
    for f in files
        p = load_meta_p(f)
        if isfinite(p)
            p = round(p; digits=3)
            if !haskey(p2path, p) || stat(f).mtime > stat(p2path[p]).mtime
                p2path[p] = f
            end
        end
    end

    avail = sort(collect(keys(p2path)))
    isempty(avail) && error("No usable meta.p found in any .jld2 files.")
    println("[grid] found p values: ", avail)

    ps_use = Float64[]
    if isempty(ps_req)
        ps_use = avail
        println("[grid] ps not provided -> plotting ALL found p values.")
    else
        ps_req_r = round.(ps_req; digits=3)
        if all(p -> haskey(p2path, p), ps_req_r)
            ps_use = ps_req_r
        else
            if !auto_nearest
                error("Some requested p are missing. Found p keys: $avail.")
            end
            mapped = map_to_nearest(ps_req_r, avail)
            println("[grid] mapping requested p -> nearest available:")
            for (a,b) in zip(ps_req_r, mapped)
                @printf("  req %.3f -> use %.3f\n", a, b)
            end
            ps_use = unique(mapped)
        end
    end

    rot_plot = rot_from_deg(rot_deg)

    turbo_ber_map = Dict{Float64,Float64}()
    jsdc_ber_map  = Dict{Float64,Float64}()

    local_use_ber_blobs = use_ber_blobs
    if local_use_ber_blobs
        if !isfile(csv_ber)
            @warn "BER CSV not found; disabling BER blobs" csv_ber
            local_use_ber_blobs = false
        else
            turbo_ber_map = load_ber_map(csv_ber, :turbo_ber)
            jsdc_ber_map  = load_ber_map(csv_ber, :jsdc_ber)
            println("[grid] BER blobs ON using: $csv_ber")
        end
    end

    fig = Figure(size=(1200, 300*length(ps_use) + 120))
    Label(fig[0, 1:3], "Constellation grid (rows=p, cols=JSDC/Turbo/EQ)", fontsize=18)

    for (ri, p0) in enumerate(ps_use)
        f = p2path[p0]
        d = JLD2.load(f)

        s_jsdc  = ComplexF64.(d["s_jsdc_all"])
        s_turbo = ComplexF64.(d["s_turbo_all"])
        s_eq    = ComplexF64.(d["s_eq_all"])

        js_ber = get(jsdc_ber_map,  p0, NaN)
        tb_ber = get(turbo_ber_map, p0, NaN)

        _panel!(fig, ri, 1, s_jsdc,  @sprintf("p=%.2f | JSDC", p0);
                edge_tol=edge_tol, rot_plot=rot_plot,
                use_ber_blob=local_use_ber_blobs && (:jsdc in ber_cols) && isfinite(js_ber),
                ber_val=isfinite(js_ber) ? js_ber : 0.1,
                ber_n=ber_n, ber_floor=ber_floor, ber_ceil=ber_ceil)

        _panel!(fig, ri, 2, s_turbo, @sprintf("p=%.2f | Turbo", p0);
                edge_tol=edge_tol, rot_plot=rot_plot,
                use_ber_blob=local_use_ber_blobs && (:turbo in ber_cols) && isfinite(tb_ber),
                ber_val=isfinite(tb_ber) ? tb_ber : 0.1,
                ber_n=ber_n, ber_floor=ber_floor, ber_ceil=ber_ceil)

_panel!(fig, ri, 3, s_eq, @sprintf("p=%.2f | EQ", p0);
        edge_tol=edge_tol, rot_plot=rot_plot,
        use_ber_blob=false,
        ber_val=0.1, ber_n=ber_n, ber_floor=ber_floor, ber_ceil=ber_ceil,
        unbox=true, unbox_eps=1e-6, auto_unbox_lim=true, unbox_q=0.995)

    end

    mkpath(dirname(out) == "" ? "." : dirname(out))
    save(out, fig)
    save(replace(out, ".png" => ".pdf"), fig)
    @info "Saved grid" out edge_tol rot_deg local_use_ber_blobs eq_unbox unbox_eps unbox_q
end

end # module
