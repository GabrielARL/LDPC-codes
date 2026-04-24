# ============================================================================
# DFEC Experiment Configuration
# ============================================================================
# This file contains all experimental parameters in one place.
# Modify these values to run different experiments.
# ============================================================================

# === Data and File Paths ===
const DATA_DIR = "data"
const RESULTS_DIR = "results"
const SIGNAL_FILE = "MFDFEC-1740453405825.txt"
const DATA_FILE = "logged_packets_and_ytrain.jld2"
const LDPC_H_FILE = "64-128-4.H"

# === Experiment Control ===
const NUM_FRAMES_TO_PROCESS = 1          # Process N frames (packets)
const PILOT_FRACTIONS = 0.36:0.05:0.46   # Pilot row fractions to test

# === Channel Estimation Parameters ===
const H_LEN = 40                    # Channel impulse response length
const K_SPARSE = 4                  # Sparsity level for OMP
const NOISE_VARIANCE = 0.01         # σ² for MMSE estimation

# === LDPC Code Parameters ===
const D_NODES = 64                  # Variable nodes
const T_NODES = 128                 # Check nodes
const NPC = 4                       # Parity check nodes per variable node
const CODE_K = 64                   # Information bits
const CODE_N = 128                  # Codeword length

# === Training Signal Parameters ===
const NUM_TRAIN = 5                 # Number of training sequences
const NUM_DATA = 20                 # Data symbols per packet
const GAP = 160                     # Gap between training and data
const FS = 24000                    # Sampling frequency (Hz)
const FC = 24000                    # Carrier frequency (Hz)
const SPSD = 24                     # Samples per symbol

# === DFE (Decision Feedback Equalizer) Parameters ===
const DFE_FILTER_LENGTH = 40        # DFE filter length (same as H_LEN)

# === Joint Decoding Parameters ===
const LAMBDA = 2.0                  # λ parameter for joint decoding
const GAMMA = 1e-3                  # γ parameter for joint decoding
const ETA = 1.0                     # η parameter for joint decoding

# === BPSK Modulation ===
const BPSK_CONSTELLATION = [1.0, -1.0]

# === Output Results ===
const RESULTS_FILE = "ldpc_ber.csv"

# ============================================================================
# Function to display configuration
# ============================================================================
function print_config()
    println("╔═══════════════════════════════════════════════════════════════╗")
    println("║         DFEC Experiment Configuration                          ║")
    println("╚═══════════════════════════════════════════════════════════════╝")
    println("🔹 Data Files:")
    println("   • Signal file: $(SIGNAL_FILE)")
    println("   • Packet data: $(DATA_FILE)")
    println("   • LDPC matrix: $(LDPC_H_FILE)")
    println()
    println("🔹 Experiment Setup:")
    println("   • Frames to process: $(NUM_FRAMES_TO_PROCESS)")
    println("   • Pilot fractions tested: $(PILOT_FRACTIONS)")
    println()
    println("🔹 LDPC Code: $(CODE_K)/$(CODE_N) (d=$(D_NODES), t=$(T_NODES))")
    println("🔹 Channel: H_len=$(H_LEN), σ²=$(NOISE_VARIANCE)")
    println("🔹 Joint Decoding: λ=$(LAMBDA), γ=$(GAMMA), η=$(ETA)")
    println()
end
