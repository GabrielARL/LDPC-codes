# lib/paths.jl
# Include-only: defines common project paths.

const ROOT     = normpath(joinpath(@__DIR__, ".."))
const LIB_DIR  = joinpath(ROOT, "lib")
const SRC_DIR  = joinpath(ROOT, "src")
const SCRIPTS  = joinpath(ROOT, "scripts")
const DATA_DIR = joinpath(ROOT, "data")
const CODES_DIR = joinpath(ROOT, "codes")

# Optional vendored LinkSim path:
const VENDOR_LINKSIM_DIR = joinpath(ROOT, "vendor", "LinkSim")
const VENDOR_LINKSIM_SRC = joinpath(VENDOR_LINKSIM_DIR, "src", "LinkSim.jl")

# Helper: load vendored LinkSim if present, else assume user does `using LinkSim`
function ensure_linksim_loaded!()
    if isdefined(Main, :LinkSim)
        return Main.LinkSim
    end

    if isfile(VENDOR_LINKSIM_SRC)
        include(VENDOR_LINKSIM_SRC)
        @info "Loaded vendored LinkSim" path=VENDOR_LINKSIM_SRC
        return Main.LinkSim
    end

    # Fall back to package dependency (must be in Project.toml / environment)
    try
        @eval Main begin
            using LinkSim
        end
        @info "Loaded LinkSim as a package dependency"
        return Main.LinkSim
    catch e
        error("LinkSim not found. Either vendor it at $(VENDOR_LINKSIM_SRC) or add it as a package dependency.\nError: $e")
    end
end

