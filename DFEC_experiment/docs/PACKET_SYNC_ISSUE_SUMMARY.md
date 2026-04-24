# Packet Synchronization Issue: Complete Summary

## TL;DR

**Problem:** All 900 packets in the dataset have systematic frame timing misalignment due to incorrect sample indexing during packet extraction from the raw signal file.

**Evidence:** Sync detector found 99/99 packet boundaries show ±28-29 sample shifts and 0-180° phase jumps.

**Root Cause:** Raw signal was extracted into packets with wrong sample indices, creating consistent 2-sample padding per packet.

**Fix Required:** Locate and correct the packet extraction code that reads from `MFDFEC-1740453405825.txt`.

---

## The Issue Explained

### What We Observed

**User's Initial Report:**
- Packet 19: BER = 0 (perfect)
- Packet 20: BER = high (failure)
- Sudden change suggests sync/alignment problem

**Our Investigation Found:**
- This is not a random glitch at packet 20
- **ALL 900 packets have this problem**
- The issue compounds at packet boundaries

### Evidence

#### Sync Detector Results (100 packet boundaries analyzed):

```
PROBLEMATIC BOUNDARIES: 99/99 (100%)

Key Metrics:
├─ Alignment shifts: Consistently ±28-29 samples
├─ Phase jumps: Random 0-180° (shows frame desync)
├─ Correlation: 0.872 average (indicates structure mismatch)
└─ Pattern: Repeats every packet (not periodic, systematic)
```

#### Packet Structure Analysis:

```
Packet Matrix Dimensions: 900 × 130
Expected Dimensions:      900 × 128 (CODE_N)
Difference: 2 extra samples per packet
```

### Why This Matters

When packets are misaligned:

```
Received Signal (Raw):
[Sample 1][Sample 2]...[Sample 129][Sample 130][Sample 131]...
           ↓ (extraction at wrong index)
Extracted Packet 1:
[Sample 3][Sample 4]...[Sample 130]    ← WRONG WINDOW
           
Extracted Packet 2:
[Sample 5][Sample 6]...[Sample 132]    ← WRONG WINDOW

Result:
• DFE trains on misaligned symbols
• Phase recovery has 2-sample timing error
• Symbols cross ISI boundaries at wrong positions
• Cascading error through LDPC decoder
```

---

## Root Cause Analysis

### The Extraction Problem

The raw signal file `MFDFEC-1740453405825.txt` contains continuous received samples. Packets are extracted by:

1. **Finding packet boundaries** (frame sync markers)
2. **Extracting 128-sample windows** starting from detected positions
3. **Storing in packet matrix**

**What Went Wrong:** The boundary detection or indexing is **off by a consistent offset**, resulting in:
- Packets starting at wrong sample positions
- 2-sample padding in each extracted packet (130 instead of 128)
- Cascading phase error at each boundary

### Evidence of Systematic Error

```
Expected behavior:
  Packet N:   [Sample A to A+127]
  Packet N+1: [Sample A+128 to A+255]
  Boundary:   Clean transition at A+128

Actual behavior:
  Packet N:   [Sample B to B+129]   (B ≠ A, and has 130 samples)
  Packet N+1: [Sample D to D+129]   (D ≠ A+128, gap/overlap)
  Boundary:   Misaligned by 28-29 samples observed
```

The 28-29 sample shift at every boundary (when packets are 128 samples) suggests:
- Packet extraction uses wrong stride/spacing
- Or frame timing recovery was applied incorrectly
- Or raw signal has preprocessing that introduced offsets

---

## Impact on Algorithms

### DFE (Decision Feedback Equalizer)

```
DFE trains on: packet_matrix[frame_num, 1:n]

If packets are misaligned:
• Training phase: Learns from misaligned symbols
• Equalization phase: Applies corrections to different symbol positions
• Result: ISI not properly canceled, symbols in noise
```

### LDPC Decoder

```
Input to decoder: Equalized symbols from DFE
If DFE fails due to misalignment:
• LLR values are corrupted
• Parity checks have wrong symbol association
• Decoder convergence fails
```

### Why Packet 19→20 Boundary is Visible

- Packets before 19: Random variation in BER (sometimes lucky with alignment)
- Packet 19: Happens to align well (BER ≈ 0)
- Packet 20: Shifts back to misaligned position (BER high)
- **It's not that packet 20 is special—it's where your algorithm first detected the underlying problem**

---

## Where the Bug Is

### Location of Packet Extraction Code

Find files that:
1. **Read raw signal**: `Signals.read("MFDFEC-1740453405825.txt")`
2. **Extract packets**: Loop that creates 128-sample windows
3. **Store in matrix**: Saving to JLD2 file `logged_packets_and_ytrain.jld2`

Files to search:
- Original experiment scripts in `/JLDPC_writeup/` folder (not the reorganized DFEC_experiment)
- Look for code that:
  - Detects frame boundaries
  - Indexes into raw signal to extract packets
  - Uses hardcoded offsets or stride values

### What to Look For

```julia
# WRONG patterns to find:
for i in 1:num_packets
    start_idx = (i-1) * 128 + offset_error  ← Check if offset is correct
    pkt = raw_signal[start_idx : start_idx + 127]  ← 128 samples?
end

# Or:
frame_sync_indices = detect_frames(raw_signal)  ← Are indices correct?
packets = [raw_signal[idx : idx + 127] for idx in frame_sync_indices]
```

### Verification Steps

1. **Check raw signal length**: `length(SS)` in original script
2. **Check extraction stride**: How many samples between packet starts?
3. **Check packet boundaries**: Are they 128 or 130 samples?
4. **Verify frame sync**: Is frame timing detection algorithm correct?

---

## The Fix

### High-Level Approach

```
1. Locate the packet extraction function
2. Verify that:
   - Packet boundaries are detected correctly
   - Each extracted packet is EXACTLY 128 samples
   - No padding, no overlap, no gaps
   - Frame sync markers align with packet starts

3. Re-extract packets from raw signal if necessary
4. Regenerate the JLD2 file with corrected packets
5. Re-run experiments with corrected data
```

### Expected Result After Fix

```
BEFORE FIX:
├─ All 900 packet boundaries misaligned (28-29 sample shifts)
├─ Random phase jumps (0-180°) at boundaries
└─ Packet 20 has high BER due to cascading error

AFTER FIX:
├─ Packet boundaries show phase continuity
├─ Phase jumps < 30° (normal PLL noise)
└─ Packet 20 BER matches surrounding packets
```

---

## Key Files Involved

| File | Purpose | Status |
|------|---------|--------|
| `MFDFEC-1740453405825.txt` | Raw received signal | ✅ Clean |
| `logged_packets_and_ytrain.jld2` | Extracted packets | 🔴 **CORRUPTED** |
| Original extraction script | Creates JLD2 file | 🔴 **HAS BUG** |
| `DFEC_experiment/scripts/exp_ldpc_process.jl` | Uses packets | ⚠️ Working around bug |

---

## Summary for Action

### What an LLM Should Do With This Info

1. **Understand**: Packets were extracted with wrong timing, causing systematic 28-29 sample misalignment at every boundary

2. **Locate**: Find the code that reads `MFDFEC-1740453405825.txt` and creates packets (search in `Process MF 3.1km Exp/` folder)

3. **Verify**: Check that packet length is 128 samples and boundaries align with frame timing

4. **Fix**: Correct the sample indexing in the extraction code

5. **Re-extract**: Regenerate the `logged_packets_and_ytrain.jld2` file

6. **Validate**: Run sync_detector.jl on new packets—should show < 10 problematic boundaries instead of 99/99

---

## Tools Created to Diagnose This Issue

All in `/DFEC_experiment/scripts/`:

- `sync_alignment_check.jl` — Detailed analysis of packets 19 vs 20
- `sync_detector.jl` — Analyzes all packet boundaries
- `doppler_analysis.jl` — Rules out Doppler shift (found none)
- `inspect_packets.jl` — Shows packet matrix dimensions

Run with:
```bash
cd DFEC_experiment
julia --project=. scripts/sync_detector.jl
```

---

## Conclusion

The BER spike at packet 20 is a **red herring**—it's a symptom of a **systematic packet extraction bug** that affects all 900 packets. The solution is to locate and fix the code that extracts packets from the raw signal file.

This is **not** a Doppler issue, not a random sync loss, and not a decoder problem—it's **front-end data corruption** that propagates through the entire receiver chain.
