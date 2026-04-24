# DFEC Experiment Framework

**Decision Feedback Equalizer + LDPC Forward Error Correction (DFEC) Decoding**

A clean, modular Julia framework for evaluating joint channel estimation and symbol decoding over simulated multicarrier wireless channels.

---

## 📋 Overview

This framework implements a complete digital receiver for BPSK-modulated signals transmitted over a wireless channel:

```
Transmitter → Channel → Receiver
                       ├─ Channel Estimation (OMP/MMSE)
                       ├─ DFE Equalization
                       ├─ SPA Decoding (baseline)
                       └─ Joint Decoding (proposed)
                            └─ Simultaneously estimates channel & decodes info bits
```

### Key Components

| Module | Purpose |
|--------|---------|
| `SignalUtils.jl` | Channel estimation, phase tracking, signal alignment |
| `LDPCJDPMemoized.jl` | LDPC code, SPA decoder, joint optimization |
| `ExperimentCore.jl` | Orchestrates pipeline, loops over parameters |
| `experiment_config.jl` | All tunable parameters in one place |
| `run_experiment.jl` | Entry point for running experiments |

---

## 🏗️ Folder Structure

```
DFEC_experiment/
├── README.md                    # This file
├── Project.toml                 # Julia package dependencies
│
├── config/
│   └── experiment_config.jl     # All parameters (modify these)
│
├── src/
│   ├── SignalUtils.jl           # Channel estimation & signal processing
│   ├── LDPCJDPMemoized.jl       # LDPC codes & decoding
│   └── ExperimentCore.jl        # Experiment orchestration
│
├── scripts/
│   └── run_experiment.jl        # Main entry point (run this)
│
├── data/
│   ├── logged_packets_and_ytrain.jld2   # Pre-recorded channel data
│   ├── MFDFEC-1740453405825.txt          # Raw signal samples
│   └── 64-128-4.{H,gen,pchk}            # LDPC code files
│
└── results/
    └── ldpc_ber.csv             # Experiment output (created after running)
```

---

## 🚀 Quick Start

### Prerequisites
- Julia 1.10+ installed
- Dependencies listed in `Project.toml` (auto-installed via `julia --project`)

### Run Experiment

```bash
cd DFEC_experiment
julia --project=. scripts/run_experiment.jl
```

This runs the experiment with default configuration and saves results to `results/ldpc_ber.csv`.

### Run The C++ Port

The repository now includes a first-pass C++ runner for the default experiment flow.

1. Export the Julia JLD2 dataset into a C++-friendly bundle:

```bash
cd DFEC_experiment
julia --project=. scripts/export_cpp_bundle.jl
```

2. Build and run the C++ experiment:

```bash
cd DFEC_experiment/cpp
make
./run_experiment_cpp ../export/cpp_bundle ../export/cpp_bundle/ldpc_ber_cpp.csv
```

This writes a CSV with the same top-level columns as the Julia experiment.

Notes:
- The C++ runner removes Julia from the runtime experiment loop, but still uses a one-time Julia export because the stored source dataset is in JLD2/DataFrame form.
- The C++ port is a practical first pass, not a bit-exact clone of every Julia detail.
- In particular, the baseline equalizer path uses a tap-based complex DFE instead of Julia's AdaptiveEstimators RLS DFE.

---

## ⚙️ Configuration

All parameters are in **`config/experiment_config.jl`**. Modify values there, then re-run the script.

### Common Parameters

```julia
const NUM_FRAMES_TO_PROCESS = 1          # Number of packets to process
const PILOT_FRACTIONS = 0.36:0.05:0.46   # Pilot symbol fractions to test
const LAMBDA = 2.0                       # Optimization weight
const GAMMA = 1e-3                       # Regularization strength
const ETA = 1.0                          # Channel estimation weight
```

**To change:**
1. Open `config/experiment_config.jl`
2. Modify constants
3. Re-run `run_experiment.jl`

---

## 📊 Understanding the Pipeline

### Step 1: Data Loading
Loads pre-recorded channel observations and packet data from JLD2 and text files.

### Step 2: Channel Estimation
Two methods estimate the wireless channel impulse response from training sequences:

- **OMP** (Orthogonal Matching Pursuit): Sparse, k-support estimation
- **MMSE**: Dense, minimum mean-squared error estimation

### Step 3: DFE Equalization
Decision Feedback Equalizer removes the channel's inter-symbol interference.

Output: `x̂` (equalized symbols)

### Step 4: SPA Decoding (Baseline)
Belief propagation decoder tries to recover information bits from `x̂`.

Output: `ber_spa` (SPA performance)

### Step 5: Joint Decoding (Proposed)
Simultaneously:
- Refines channel estimate
- Decodes information bits
- Optimizes via Optim.jl

Output: `ber_grad` (joint decoding performance)

### Step 6: Metrics
For each frame, computes:

```
ber_dfe     = DFE error rate (no LDPC)
ber_spa     = SPA error rate (no joint estimation)
ber_grad    = Joint decoding error rate (proposed)
ber_min     = min(ber_spa, ber_dfe) = best baseline
```

---

## 📈 Output Format

Results are saved to `results/ldpc_ber.csv`:

| frame | block | ber_grad | ber_dfe | ber_spa | ber_min | pilot_frac |
|-------|-------|----------|---------|---------|---------|------------|
| 1     | 1     | 0.0      | 0.1562  | 0.2     | 0.1562  | 0.36       |
| 1     | 1     | 0.0      | 0.1562  | 0.1951  | 0.1562  | 0.41       |
| 1     | 1     | 0.0      | 0.1016  | 0.2059  | 0.1016  | 0.46       |

**Interpretation:**
- Lower BER is better
- Compare `ber_grad` (proposed) to `ber_min` (baseline)
- Analyze how performance varies with `pilot_frac`

---

## 🔍 Module Documentation

### `SignalUtils.jl`

Core signal processing functions:

```julia
# Channel Estimation
estimate_omp_channel(y, x, L_h, k)      # OMP estimation
estimate_mmse_channel(y, x, L_h; σ²)    # MMSE estimation

# Phase Correction
argminphase(x̂, x)                       # Find phase rotation minimizing BER
correct_bpsk_phase_shift(...)           # Full phase correction pipeline

# Utilities
stdize(signal)                           # Standardize to unit variance
track_bpsk_carrier_pll(...)             # PLL-based demodulation
```

### `LDPCJDPMemoized.jl`

LDPC code and decoding:

```julia
# Code initialization
initcode(d_nodes, t_nodes, npc; pilot_row_fraction)

# Encoding
encode(code, bits)

# Decoding
sum_product_decode(H_sparse, llr, ...)  # SPA decoder
decode_sparse_joint(y, code, ...)       # Joint channel + symbol decoding

# Utilities
modulate(bit; θ)                        # Map {0,1} → {±1} × e^{jθ}
demodulate(symbol; θ)                  # Inverse modulation
```

### `ExperimentCore.jl`

Orchestration and pipeline:

```julia
run_dfec_experiment(config_file)        # Main entry point

# Internal functions
load_data(dir, file, ...)               # Load JLD2/signal files
estimate_channels(y, x, ...)            # Run channel estimation
process_pilot_fraction(cache, pilot_frac, ...) # Process one config
```

---

## 🛠️ Extending the Framework

### Add a New Decoder

1. Implement in `src/MyDecoder.jl`:
```julia
module MyDecoder
function my_decoder(y, code, ...)
    # Your decoder here
    return bits
end
end
```

2. Import in `run_experiment.jl`:
```julia
include("../src/MyDecoder.jl")
using .MyDecoder
```

3. Call in `ExperimentCore.jl`:
```julia
result = MyDecoder.my_decoder(y_data, code, ...)
```

### Add New Metrics

Modify `process_pilot_fraction()` in `ExperimentCore.jl` to compute and return additional metrics in the results DataFrame.

### Run Batch Experiments

Create `scripts/batch_experiment.jl`:
```julia
for noise_level in [0.01, 0.02, 0.05]
    σ² = noise_level
    run_dfec_experiment("../config/experiment_config.jl")
end
```

---

## 🐛 Troubleshooting

**Error: `Package X not found`**
```bash
cd DFEC_experiment
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

**Memory issues with large datasets**
- Reduce `NUM_FRAMES_TO_PROCESS` in config
- Reduce size of `y_train_matrix` preprocessing

**Results don't match expected**
- Check all parameters in `config/experiment_config.jl`
- Verify data files exist in `data/`
- Ensure LDPC code matrices (`.H`, `.gen`) are correct

---

## 📚 References

### Algorithm Overview
1. **Channel Estimation:** OMP (sparse), MMSE (dense)
2. **Equalization:** DFE with RLS adaptation
3. **Decoding:** SPA vs Joint channel-symbol optimization via L-BFGS
4. **Code:** LDPC (64,128) with variable pilot fractions

### Key Parameters
- **λ, γ, η:** Control tradeoff between channel fit, sparsity, and symbol likelihood
- **Pilot fraction:** Fraction of transmitted symbols known at receiver for reference
- **k_sparse:** Number of dominant taps in channel estimate

---

## 📝 License & Attribution

This framework reorganizes experimental code for clarity and modularity. 

Original code structure from MF 3.1km experiment.
Refactored for pedagogical purposes: clear separation of concerns, self-documenting modules, reproducible configuration.

---

## ✉️ Questions?

Refer to:
- Function docstrings (start of each function)
- Config comments in `experiment_config.jl`
- This README's module documentation section
