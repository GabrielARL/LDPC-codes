#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

REC="data/raw/logged_packets_and_ytrain.jld2"
OUTDIR="data"
SEED="626374780"          # 0x2565121024 in decimal (fixed π512)
# If you prefer, change SEED to any integer for a different π512.

julia --project=. scripts/make_replayswap_bases_from_raw.jl \
  --rec "$REC" \
  --outdir "$OUTDIR" \
  --seed "$SEED"

echo
echo "Verify interleaver flags:"
julia --project=. scripts/verify_interleaver.jl \
  --in "data/replayswap_qpsk_concat_256_512_1024_from_realdata_donorLS_h20_rho1e-2.jld2"
julia --project=. scripts/verify_interleaver.jl \
  --in "data/replayswap_qpsk_RSCconcat_256_512_1024_from_realdata_donorLS_h20_rho1e-2.jld2"
