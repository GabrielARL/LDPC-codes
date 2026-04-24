#!/usr/bin/env julia
# scripts/merge_rsc_turbo_into_raw_tidy_csv.jl
#
# Append RSC TurboEQ mean rows (grouped by p) into RAW tidy CSV.
#
# RAW tidy schema:
#   pilot_frac, method, ber, psr_pkt, psr64, nframes, lam_pil, agree_pilots
#
# Default mapping (RSC -> tidy):
#   pilot_frac = p
#   method     = "Turbo"
#   ber        = mean(b128_post_ber)
#   psr_pkt    = mean(b128_post_psr)
#   psr64      = mean(u64_psr)
#   nframes    = count rows for that p
#   lam_pil    = NaN
#   agree_pilots = 0
#
# Usage:
#   julia --project=. scripts/merge_rsc_turbo_into_raw_tidy_csv.jl
#   julia --project=. scripts/merge_rsc_turbo_into_raw_tidy_csv.jl --method "Turbo"
#   julia --project=. scripts/merge_rsc_turbo_into_raw_tidy_csv.jl --out_csv data/runs/raw_plus_rsc.csv
#
# Overrides:
#   --raw_csv <path>
#   --rsc_csv <path>
#   --out_csv <path>   (default overwrites raw_csv)
#   --replace 1|0      (remove existing rows with same (pilot_frac, method), default 1)
#   --ber_col <name>       (default b128_post_ber)
#   --psr_pkt_col <name>   (default b128_post_psr)
#   --psr64_col <name>     (default u64_psr)

using CSV, DataFrames, Statistics, Printf

parse_int(s::String) = parse(Int, strip(s))

# Normalize column header strings:
# - strip whitespace/tabs/newlines
# - remove UTF-8 BOM (\ufeff)
# - replace weird nonbreaking spaces
function normalize_header(s::String)
    s2 = replace(s, '\ufeff' => "")          # BOM
    s2 = replace(s2, '\u00a0' => ' ')        # NBSP -> space
    s2 = strip(s2)
    return s2
end

function sanitize_colnames!(df::DataFrame)
    old = names(df)  # Vector{String}
    new = normalize_header.(old)
    if any(old .!= new)
        rename!(df, Pair.(old, new))
    end
    return df
end

# Get Symbol column if present (after sanitize), else error with helpful list
function require_col!(df::DataFrame, colname::String; label::String="")
    s = Symbol(colname)
    if !(s in propertynames(df))
        error("$(label)CSV missing column \"$colname\". Columns=$(names(df))")
    end
    return s
end

function main()
    raw_csv = "data/runs/raw_dfec_oraclepilots_psweep.csv"
    rsc_csv = "data/runs/psr_bpsk_rsc_turbo.csv"
    out_csv = ""  # overwrite raw_csv if empty
    method  = "Turbo"
    replace_existing = true

    ber_col = "b128_post_ber"
    psr_pkt_col = "b128_post_psr"
    psr64_col = "u64_psr"

    # CLI
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--raw_csv"; i+=1; raw_csv = ARGS[i]
        elseif a == "--rsc_csv"; i+=1; rsc_csv = ARGS[i]
        elseif a == "--out_csv"; i+=1; out_csv = ARGS[i]
        elseif a == "--method"; i+=1; method = ARGS[i]
        elseif a == "--replace"; i+=1; replace_existing = (parse_int(ARGS[i]) != 0)
        elseif a == "--ber_col"; i+=1; ber_col = ARGS[i]
        elseif a == "--psr_pkt_col"; i+=1; psr_pkt_col = ARGS[i]
        elseif a == "--psr64_col"; i+=1; psr64_col = ARGS[i]
        elseif a == "--help" || a == "-h"
            println("""
Usage:
  julia --project=. scripts/merge_rsc_turbo_into_raw_tidy_csv.jl [args]

Args:
  --raw_csv <path>        (default data/runs/raw_dfec_oraclepilots_psweep.csv)
  --rsc_csv <path>        (default data/runs/psr_bpsk_rsc_turbo.csv)
  --out_csv <path>        (default overwrite raw_csv)
  --method <string>       (default "Turbo")
  --replace 1|0           (default 1)

Mapping:
  --ber_col <name>        (default b128_post_ber)
  --psr_pkt_col <name>    (default b128_post_psr)
  --psr64_col <name>      (default u64_psr)
""")
            return
        else
            error("Unknown arg: $a")
        end
        i += 1
    end

    isempty(out_csv) && (out_csv = raw_csv)

    isfile(raw_csv) || error("Missing RAW tidy CSV: $raw_csv")
    isfile(rsc_csv) || error("Missing RSC CSV: $rsc_csv")

    df_raw = CSV.read(raw_csv, DataFrame)
    df_rsc = CSV.read(rsc_csv, DataFrame)

    sanitize_colnames!(df_raw)
    sanitize_colnames!(df_rsc)

    # Required RAW columns
    c_pilot_frac = require_col!(df_raw, "pilot_frac"; label="RAW ")
    c_method_raw = require_col!(df_raw, "method"; label="RAW ")

    # Required RSC columns (robust header handling)
    c_p = require_col!(df_rsc, "p"; label="RSC ")
    c_ber = require_col!(df_rsc, ber_col; label="RSC ")
    c_psr_pkt = require_col!(df_rsc, psr_pkt_col; label="RSC ")
    c_psr64 = require_col!(df_rsc, psr64_col; label="RSC ")

    # Build RSC mean rows per p
    df_add = DataFrame(
        pilot_frac = Float64[],
        method     = String[],
        ber        = Float64[],
        psr_pkt    = Float64[],
        psr64      = Float64[],
        nframes    = Int[],
        lam_pil    = Float64[],
        agree_pilots = Int[]
    )

    g = groupby(df_rsc, c_p)

    for sub in g
        pval = Float64(first(sub[!, c_p]))
        nframes = nrow(sub)

        ber_mean     = mean(skipmissing(sub[!, c_ber]))
        psr_pkt_mean = mean(skipmissing(sub[!, c_psr_pkt]))
        psr64_mean   = mean(skipmissing(sub[!, c_psr64]))

        push!(df_add, (
            pilot_frac = pval,
            method = method,
            ber = Float64(ber_mean),
            psr_pkt = Float64(psr_pkt_mean),
            psr64 = Float64(psr64_mean),
            nframes = nframes,
            lam_pil = NaN,
            agree_pilots = 0
        ))
    end

    # Debug prints (so we can’t “quietly” add nothing)
    ps_found = sort(unique(Float64.(df_rsc[!, c_p])))
    println("RSC p values found: ", ps_found)
    @printf("RSC rows total: %d | groups: %d | rows to add: %d (method=%s)\n",
            nrow(df_rsc), length(g), nrow(df_add), method)

    if nrow(df_add) == 0
        error("No rows to add. This means grouping by p produced no groups (unexpected). Check RSC CSV headers/contents.")
    end

    # Normalize RAW types
    df_raw[!, c_pilot_frac] = Float64.(df_raw[!, c_pilot_frac])
    df_raw[!, c_method_raw] = String.(df_raw[!, c_method_raw])

    # Optionally drop existing (pilot_frac, method) duplicates
    if replace_existing
        add_keys = Set([(round(r.pilot_frac; digits=12), r.method) for r in eachrow(df_add)])
        keep = trues(nrow(df_raw))
        for (idx, r) in enumerate(eachrow(df_raw))
            k = (round(Float64(r[c_pilot_frac]); digits=12), String(r[c_method_raw]))
            if k in add_keys
                keep[idx] = false
            end
        end
        df_raw = df_raw[keep, :]
    end

    df_out = vcat(df_raw, df_add; cols=:union)

    # Ensure tidy column order
    tidy_cols = [:pilot_frac, :method, :ber, :psr_pkt, :psr64, :nframes, :lam_pil, :agree_pilots]
    for c in tidy_cols
        (c in propertynames(df_out)) || (df_out[!, c] = missing)
    end
    extra_cols = [c for c in propertynames(df_out) if !(c in tidy_cols)]
    df_out = df_out[!, vcat(tidy_cols, extra_cols)]

    sort!(df_out, [:pilot_frac, :method])

    mkpath(dirname(out_csv))
    CSV.write(out_csv, df_out)
    println("Wrote merged tidy CSV → $out_csv (rows=$(nrow(df_out)))")
end

main()
