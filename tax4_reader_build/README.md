# TAx4 std-recon reader with `nfold` (for the vlen adapter)

`dst_adapter_tax4_vlen` needs a per-hit `rufptn_.nfold` to map waveforms to
hits unambiguously (see below). The stock TAx4 std-recon reader
(`sditerator_add_standard_recon.run` in the ceph TALE install) **does not print
`nfold`**, so we rebuild the reader from a single modified source.

## What's different from the stock reader

Only two additions, both in `src/sditerator_cppanalysis_add_standard_recon_nfold.cpp`
(a copy of the ceph `sditerator_cppanalysis_add_standard_recon.cpp`):

1. **`#SD meta DATA`**: appends `rufptn_.nfold[x]` → **12 fields/hit** (was 11).
2. **`#EVENT DATA`**: appends the free-curvature geometry fit + curvature
   (`rusdgeom_.theta[2] phi[2] dtheta[2] dphi[2] chi2[2] ndof[2] t0[2] dt0[2]
   a da`) → **42 fields** (the on-disk ceph source printed 32; the *installed*
   ceph binary already emitted these 42 — verified byte-identical, max diff 0).

## Why `nfold` is required

The waveform stream follows hit order, but when a detector appears at adjacent
hit positions with extra waveforms, the waveform→hit split is ambiguous from
the printed data alone (folds are **not** reliably contiguous — a single hit
can have waveforms 428 clkcnt apart; clkcnt gaps form a continuum with no clean
threshold). Inferring the split greedily lost the time-trace of ~3.0% of good
hits (18.5% of events). With `nfold` present, `repeat(swap(xxyy), nfold) ==
wf_xxyy` exactly (934/934 events checked), i.e. 0% trace loss.

## Build

```bash
bash build.sh
```

Reads all headers/libs/ROOT from the ceph install **READ-ONLY** — nothing on
ceph is modified. Output binary: `$TAX4_READER_OUT/bin/sditerator_add_standard_recon_nfold.run`
(default `/home/antonpr/local_sdanalysis/tax4_reader/`). Keep that path in sync
with `dstparser/paths.py::dst_reader_tax4_std_recon_nfold`.

Overridable env vars: `TAX4_SDANALYSIS`, `TAX4_OPENSSL10`, `TAX4_ROOT_SH`,
`TAX4_READER_OUT`.

### Toolchain notes (Rocky Linux)
- `g++` + CERN ROOT (sourced from ceph). Default compiler is `g++` (Intel only
  if `usingicc` is set).
- ROOT's `libNet` needs legacy `libssl.so.10`/`libcrypto.so.10` → linked from
  the benMC `openssl10` dir (same libs dstparser already puts on
  `LD_LIBRARY_PATH` at runtime for Rocky).
- `libbz2` is linked by soname from `/usr/lib64/libbz2.so.1` so it resolves
  system-wide at runtime (no `LD_LIBRARY_PATH` needed for it).

The built binary is intentionally **not** committed (binaries don't belong in
git). Rebuild with `build.sh` on any new machine.
