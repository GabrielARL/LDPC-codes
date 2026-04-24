#!/usr/bin/env julia
# scripts/c3_grid.jl
#
# Constellation grid (rows=p, cols=JSDC/Turbo/EQ)
#
# Usage:
#   julia --project=. scripts/c3_grid.jl
#   julia --project=. scripts/c3_grid.jl data/runs_comp3ways_constellations "" data/plots/c3_grid.png
#   julia --project=. scripts/c3_grid.jl data/runs_comp3ways_constellations "" data/plots/c3_grid_rot90.png --rot_deg 90

# Avoid module redefinition warnings in the REPL if you re-run includes:
if !isdefined(Main, :C3GridUtils)
    include(joinpath(@__DIR__, "c3_grid_utils.jl"))
end
using .C3GridUtils

# ---------------- CLI ----------------
input = length(ARGS) >= 1 ? ARGS[1] : joinpath("data", "runs_comp3ways_constellations")
ps_arg = length(ARGS) >= 2 ? ARGS[2] : ""
ps_req = isempty(strip(ps_arg)) ? Float64[] : C3GridUtils.parse_ps(ps_arg)
out    = length(ARGS) >= 3 ? ARGS[3] : joinpath("data", "plots", "c3_grid.png")

edge_tol = 0.05
auto_nearest = true
rot_deg = 0

# defaults: ON
csv_ber = joinpath("data","runs","compare_c3_psweep.csv")
use_ber_blobs = true
ber_n = 80000
ber_floor = 1e-4
ber_ceil  = 0.49
ber_cols = Set([:jsdc, :turbo])

i = 4
while i <= length(ARGS)
    a = ARGS[i]
    if a == "--edge_tol"; i += 1; edge_tol = parse(Float64, ARGS[i])
    elseif a == "--auto_nearest"; i += 1; auto_nearest = (parse(Int, ARGS[i]) != 0)
    elseif a == "--rot_deg"; i += 1; rot_deg = parse(Int, ARGS[i])

    elseif a == "--csv_ber"; i += 1; csv_ber = ARGS[i]
    elseif a == "--blobs_from_ber"; i += 1; use_ber_blobs = (parse(Int, ARGS[i]) != 0)
    elseif a == "--ber_n"; i += 1; ber_n = parse(Int, ARGS[i])
    elseif a == "--ber_floor"; i += 1; ber_floor = parse(Float64, ARGS[i])
    elseif a == "--ber_ceil"; i += 1; ber_ceil = parse(Float64, ARGS[i])
    elseif a == "--ber_cols"; i += 1; ber_cols = C3GridUtils.parse_cols(ARGS[i])

    else
        error("Unknown arg: $a")
    end
    i += 1
end

C3GridUtils.main(input=input, ps_req=ps_req, out=out,
                 edge_tol=edge_tol, auto_nearest=auto_nearest,
                 rot_deg=rot_deg,
                 csv_ber=csv_ber, use_ber_blobs=use_ber_blobs,
                 ber_n=ber_n, ber_floor=ber_floor, ber_ceil=ber_ceil,
                 ber_cols=ber_cols)
