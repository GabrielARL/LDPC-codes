"""
    inspect_packets.jl

Inspect packet structure and extraction in the JLD2 file.
"""

push!(LOAD_PATH, joinpath(@__DIR__, "..", "config"))
include("../config/experiment_config.jl")

using JLD2, DataFrames, Statistics, Printf

# Load data
data_dir = DATA_DIR
if !isabspath(data_dir)
    project_root = dirname(@__DIR__)
    data_dir = joinpath(project_root, data_dir)
end

jld2_path = joinpath(data_dir, DATA_FILE)
@load jld2_path all_ytrain_df all_packets_df

packet_matrix = Matrix(select(all_packets_df, Not([:frame, :block])))

println("="^80)
println("📋 PACKET STRUCTURE INSPECTION")
println("="^80 * "\n")

println("Packet Matrix Properties:")
println("├─ Shape: $(size(packet_matrix))")
println("├─ Data type: $(eltype(packet_matrix))")

packet_abs = abs.(packet_matrix)
println("├─ Min magnitude: $(round(minimum(packet_abs), digits=4))")
println("├─ Max magnitude: $(round(maximum(packet_abs), digits=4))")
println("└─ Memory: $(round(sizeof(packet_matrix)/1e6, digits=1)) MB\n")

num_packets = size(packet_matrix, 1)
packet_len = size(packet_matrix, 2)

println("Code Parameters:")
println("├─ CODE_N (codeword length): $(CODE_N)")
println("├─ Actual packet_matrix columns: $(packet_len)")
println("├─ Ratio (cols/CODE_N): $(round(packet_len/CODE_N, digits=3))")
println("└─ Potential packet stride: $(packet_len)\n")

# Analyze first few packets
println("First 5 Packets Energy Profile:")
println("┌────────┬────────────┬────────────┬────────────┬──────────────┐")
println("│ Pkt    │ Mean Pow   │ Min        │ Max        │ Std Dev      │")
println("├────────┼────────────┼────────────┼────────────┼──────────────┤")

for i in 1:min(5, num_packets)
    pkt = packet_matrix[i, :]
    mean_pow = mean(abs2, pkt)
    pkt_abs = abs.(pkt)
    min_val = minimum(pkt_abs)
    max_val = maximum(pkt_abs)
    std_val = std(pkt)
    @printf "│ %6d │ %10.4f │ %10.4f │ %10.4f │ %12.4f │\n" i mean_pow min_val max_val std_val
end
println("└────────┴────────────┴────────────┴────────────┴──────────────┘\n")

# Check for patterns at boundaries
println("Energy at Packet Boundaries:")
println("(Checking if packets are continuous or padded)\n")

println("┌──────────┬──────────────┬──────────────┬──────────────┬──────────┐")
println("│ Boundary │ Last(Pkt N)  │ Gap before   │ First(Pkt N) │ Gap size │")
println("│          │ Min/Max      │ (zeros?)     │ +1 Min/Max   │ samples? │")
println("├──────────┼──────────────┼──────────────┼──────────────┼──────────┤")

for i in 1:min(10, num_packets-1)
    pkt_n = packet_matrix[i, :]
    pkt_n1 = packet_matrix[i+1, :]

    pkt_n_end_abs = abs.(pkt_n[max(1, end-5):end])
    pkt_n1_start_abs = abs.(pkt_n1[1:min(5, length(pkt_n1))])

    min_last_n = minimum(pkt_n_end_abs)
    max_last_n = maximum(pkt_n_end_abs)

    min_first_n1 = minimum(pkt_n1_start_abs)
    max_first_n1 = maximum(pkt_n1_start_abs)

    # Check if there's a gap (zeros between packets)
    gap_check = "No"
    gap_size = 0

    @printf "│ %3d→%3d  │ [%7.3f,%7.3f] │ %-12s │ [%7.3f,%7.3f] │ %8d │\n" \
        i i+1 min_last_n max_last_n gap_check min_first_n1 max_first_n1 gap_size
end
println("└──────────┴──────────────┴──────────────┴──────────────┴──────────┘\n")

# Check packet periodicity
println("Packet Length Analysis:")

# Get first column of packet matrix to check if there's a pattern
first_col_vals = packet_matrix[:, 1]
println("├─ First column statistics:")
println("│  ├─ Mean: $(round(mean(first_col_vals), digits=4))")
println("│  ├─ Std: $(round(std(first_col_vals), digits=4))")
println("│  ├─ Min: $(round(minimum(first_col_vals), digits=4))")
println("│  └─ Max: $(round(maximum(first_col_vals), digits=4))")

# Check if packets are repeating
energy_vals = [mean(abs2, packet_matrix[i, :]) for i in 1:num_packets]
println("├─ Packet energy pattern:")
println("│  ├─ Mean energy: $(round(mean(energy_vals), digits=4))")
println("│  ├─ Std energy: $(round(std(energy_vals), digits=4))")
println("│  └─ Max-min energy ratio: $(round(maximum(energy_vals)/minimum(energy_vals), digits=3))")

# Check for 45-packet repetition (since code uses `repeat(x_datas,45,1)`)
if num_packets >= 90
    energy_1_45 = energy_vals[1:45]
    energy_46_90 = energy_vals[46:90]
    corr_45 = cor(energy_1_45, energy_46_90)
    println("├─ Energy correlation at 45-packet period: $(round(corr_45, digits=4))")
    println("│  └─ Suggests data repeats every 45 packets: $(abs(corr_45) > 0.8 ? "YES" : "NO")")
end

println("\n" * "="^80)
println("🔍 HYPOTHESIS: Packet Extraction Issue")
println("="^80 * "\n")

if packet_len == CODE_N
    println("✅ Packet length matches CODE_N (128 samples)")
    println("   Packets appear correctly sized.")
elseif packet_len > CODE_N
    println("⚠️  Packet length ($(packet_len)) > CODE_N ($(CODE_N))")
    println("   Extra: $(packet_len - CODE_N) samples per packet")
    println("   Possible causes:")
    println("   • Packets include padding/guard intervals")
    println("   • Packets overlap by $(packet_len - CODE_N) samples")
    println("   • Raw signal not properly decimated/extracted")
else
    println("⚠️  Packet length ($(packet_len)) < CODE_N ($(CODE_N))")
    println("   Missing: $(CODE_N - packet_len) samples per packet")
end

# The 28-29 sample shift finding
println("\n🔴 THE 28-29 SAMPLE SHIFT:")
println("   Every packet boundary shows ±28-29 sample misalignment")
println("   This is ~23% of packet length ($(round(100*28/CODE_N, digits=1))%)")
println("   Pattern: $(round(28 * 45 / 128, digits=1)) × CODE_N shift per 45 packets")

println("\n" * "="^80)
