#!/bin/bash


if [ "x${BASH_ARGV[0]}" = "x" ]; then
    if [ ! -f ./sdanalysis_env.sh ]; then
        echo ERROR: must "cd where/sdanalysis/is" before calling "./sdanalysis_env.sh" for this version of bash!
        SDDIR=; export SDDIR
        return
    fi
    SDDIR="$PWD"; export SDDIR
else
    # get param to "."
    THIS=$(dirname ${BASH_ARGV[0]})
    SDDIR=$(cd ${THIS};pwd); export SDDIR
fi
export PATH=$SDDIR/bin":"$PATH
export LD_LIBRARY_PATH=$SDDIR/lib:$LD_LIBRARY_PATH

export PATH=/disk/ta/data/user/ekido/PCRE:$PATH
export LD_LIBRARY_PATH=/disk/ta/data/user/ekido/PCRE:$LD_LIBRARY_PATH

#source /ta/work/user/hyomin/root5.34/bin/thisroot.sh
#source /ceph/sharedfs/users/f/fujisue/root/bin/thisroot.sh
source /ceph/work/SATORI/projects/TA-ASIoP/root/bin/thisroot.sh
