#!/bin/bash
# Build a TAx4 sditerator_add_standard_recon reader that ALSO emits
# rufptn_.nfold (12-field #SD meta DATA) and the free-curvature geometry fit
# (42-field #EVENT DATA). Only the ONE modified cppanalysis source (in this
# repo, src/ below) is ours; every header/library/ROOT dependency is read from
# the ceph sdanalysis install READ-ONLY. Nothing on ceph is written.
#
# Output goes to $TAX4_READER_OUT (default below). dstparser expects the built
# binary at paths.dst_reader_tax4_std_recon_nfold -- keep them in sync.
set -e

CEPH=${TAX4_SDANALYSIS:-/ceph/work/SATORI/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM}
OPENSSL10=${TAX4_OPENSSL10:-/ceph/work/SATORI/projects/TA-ASIoP/benMC/libs_rocky_linux/openssl10}  # ROOT libNet needs legacy libssl.so.10 (read-only)
ROOT_SH=${TAX4_ROOT_SH:-/ceph/work/SATORI/projects/TA-ASIoP/root/bin/thisroot.sh}
OUT=${TAX4_READER_OUT:-/home/antonpr/local_sdanalysis/tax4_reader}

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/src"

source "$ROOT_SH"
ROOTCFLAGS=$(root-config --cflags)
ROOTLIBS=$(root-config --libs)

INCS="-I$CEPH/dst2k-ta/inc -I$CEPH/inc -I$CEPH/sdfdrt -I$CEPH/sditerator/inc"
CPPFLAGS="$ROOTCFLAGS -O3 -Wall -fPIC"

mkdir -p "$OUT/obj" "$OUT/bin"

# our modified analysis (this repo) + main + util (read from ceph, unmodified)
g++ $CPPFLAGS $INCS -c "$SRC_DIR/sditerator_cppanalysis_add_standard_recon_nfold.cpp" -o "$OUT/obj/cppanalysis.o"
g++ $CPPFLAGS $INCS -c "$CEPH/sditerator/src/sditerator_add_standard_recon.cpp"       -o "$OUT/obj/main.o"
g++ $CPPFLAGS $INCS -c "$CEPH/sditerator/src/sditerator_util.cpp"                     -o "$OUT/obj/util.o"

g++ -O3 "$OUT/obj/cppanalysis.o" "$OUT/obj/main.o" "$OUT/obj/util.o" \
    -L"$CEPH/lib" -lsden -lsduti \
    $ROOTLIBS -lMinuit -lSpectrum \
    -L"$CEPH/dst2k-ta/lib" -ldst2k \
    -lm -lc -lz -L/usr/lib64 -l:libbz2.so.1 \
    -L"$OPENSSL10" -l:libssl.so.10 -l:libcrypto.so.10 \
    -o "$OUT/bin/sditerator_add_standard_recon_nfold.run"

echo "BUILD_OK -> $OUT/bin/sditerator_add_standard_recon_nfold.run"
ls -la "$OUT/bin/sditerator_add_standard_recon_nfold.run"
