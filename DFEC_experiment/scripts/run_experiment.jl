"""
    run_experiment.jl

DFEC (Decision Feedback Equalizer + LDPC FEC) Experiment

This script runs the complete DFEC decoding pipeline:
  1. Load simulated received signals and training data
  2. Estimate wireless channel using OMP and MMSE methods
  3. Apply DFE to equalize the channel
  4. Decode information bits using:
     - Sum-Product Algorithm (SPA) - LDPC decoding only
     - Joint Channel and Symbol Estimation (DFEC) - proposed method
  5. Measure and compare BER performance

Configuration: Modify config/experiment_config.jl to change parameters

Usage:
    julia --project=.. run_experiment.jl

Output:
    Results are saved to results/ldpc_ber.csv
"""

# Add project modules to path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "config"))

# Load configuration and modules
include("../config/experiment_config.jl")
include("../src/SignalUtils.jl")
include("../src/LDPCJDPMemoized.jl")
include("../src/ExperimentCore.jl")

using .ExperimentCore
using .LDPCJDPMemoized
using .SignalUtils
using Statistics

# ============================================================================
# Main Entry Point
# ============================================================================
function main()
    try
        # Run the experiment
        results = ExperimentCore.run_dfec_experiment("../config/experiment_config.jl")

        # Display summary
        println("\n📊 Results Summary:")
        println("├─ Packet rows processed: $(size(results, 1))")
        println("├─ Unique frames touched: $(length(unique(results.frame)))")
        println("├─ Pilot fractions tested: $(length(unique(results.pilot_frac)))")
        println("├─ Mean DFEC BER: $(round(mean(results.ber_grad), digits=4))")
        println("├─ Mean DFE+FEC BER: $(round(mean(results.ber_min), digits=4))")
        println("└─ Improvement: $(round(mean(results.ber_dfe) - mean(results.ber_min), digits=4))")

        return results

    catch e
        println("\n❌ Error occurred: $(e)")
        println("\nStack trace:")
        showerror(stderr, e, catch_backtrace())
        return nothing
    end
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
