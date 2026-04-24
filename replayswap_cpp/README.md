# replayswap_cpp

This folder is the C++ runtime port for the active `replayswap` BPSK 1/2 experiment path.

It is designed to work with a binary bundle exported from the original Julia/JLD2 data:

```bash
cd /home/gabiel/Documents/GitHub/BP/LDPC-codes
julia --project=replayswap --startup-file=no replayswap/scripts/export_cpp_bundle.jl

cd replayswap_cpp
make
./run_replayswap_cpp
```

Useful modes:

```bash
./run_replayswap_cpp --raw_only
./run_replayswap_cpp --rsc_only
./run_replayswap_cpp --ps 0.0:0.1:0.5 --nperp 20 --nblk 200
./run_replayswap_cpp --bundle export/default_bundle
```

Outputs default to the bundle directory:

- `raw_dfec_oraclepilots_psweep_cpp.csv`
- `psr_bpsk_rsc_turbo_cpp.csv`

Current scope:

- Ports the main `replayswap/scripts/apsr_bpsk_1_2_psrbar.jl` runtime path.
- Exports the raw and RSC datasets into a C++-readable bundle.
- Leaves historical plotting and `scripts/temp/*.jl` utilities out of the first-pass port.
