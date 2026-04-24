#!/usr/bin/env julia
using JLD2

function main()
    inpath = ""
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--in"
            i += 1
            inpath = ARGS[i]
        else
            error("Usage: julia --project=. scripts/verify_interleaver.jl --in <dataset.jld2>")
        end
        i += 1
    end
    isempty(inpath) && error("Missing --in <dataset.jld2>")
    isfile(inpath) || error("No file: $inpath")

    d = JLD2.load(inpath)
    meta = haskey(d, "meta_out") ? d["meta_out"] : nothing

    println("File: $inpath")
    if meta === nothing
        println("  meta_out: MISSING")
        return
    end

    if meta isa NamedTuple && hasproperty(meta, :interleaver)
        it = getproperty(meta, :interleaver)
        enabled = (it isa NamedTuple && hasproperty(it, :enabled)) ? it.enabled : false
        println("  interleaver.enabled = ", enabled)
        if enabled
            has_pi    = (it isa NamedTuple && hasproperty(it, :π512) && !isempty(it.π512))
            has_piinv = (it isa NamedTuple && hasproperty(it, :π512_inv) && !isempty(it.π512_inv))
            println("  has π512     = ", has_pi)
            println("  has π512_inv = ", has_piinv)
        end
    else
        println("  interleaver: MISSING in meta_out")
    end
end

main()
