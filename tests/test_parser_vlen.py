import numpy as np
from dstparser import parse_dst_file_vlen
from time import time
from dstparser.cli.cli import parse_config
import sys


def test_parser(dst_file, print_read_data=False, add_xmax=False):

    if len(sys.argv) > 1:
        config = parse_config(sys.argv[1])
    else:
        config = None
    start = time()

    # The only function that are required to parsing
    # is parse_dst_file
    # paths to the directories with data is dstparser.paths module
    # data = parse_dst_file(dst_file, up_low_traces=True)

    xmax_reader = None
    if add_xmax:
        from dstparser.xmax_reader import XmaxReader

        xmax_dir = "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04proton/080417_160603/Em1_bsdinfo"
        xmax_reader = XmaxReader(xmax_dir, "**/DAT*_xmax.txt", "QGSJetII-04")

    data = parse_dst_file_vlen(
        dst_file,
        xmax_reader=xmax_reader,
        add_shower_params=True,
        add_standard_recon=True,
        config=config,
    )

    end = time()
    # print(data)

    if print_read_data:
        print(f"\nConverted arrays:\n---")
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                print(key, val.shape)
            else:
                print(key, len(val), val)

    if config is not None:
        print(f'event_id = {data["id_event"]}')
        print(f'corsika_shower_id = {data["id_corsika_shower"]}')
        print(f'energy_bin_id = {data["id_energy_bin"]}')
        print(f'data_set_id = {data["id_data_set"]}')

    print("----")
    print(f"Parse time: {end - start:.3f} sec")


if __name__ == "__main__":
    # dst_file = (
    #     "/ceph/work/SATORI/projects/TA-ASIoP/INR_group/cluster82/grisha/tasdmc_SIBYLL_fe/p2/"
    #     "DAT013520.corsika77420.SIBYLL.tar.gz.spctr1.1945.noCuts.dst.gz"
    # )
    # dst_file = "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04proton/080417_160603/Em1_bsdinfo/DAT055402_gea.rufldf.dst.gz"
    dst_file = "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04proton/080417_160603/Em1_bsdinfo/DAT051419_gea.rufldf.dst.gz"
    # dst_file = "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/tax4/qgsii04proton/north/221101to240124/DAT010019_gea.rufldf.dst.gz"
    # dst_file = "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04nitrogen/080417_160603/Em1_bsdinfo/DAT081325_gea.rufldf.dst.gz"

    test_parser(dst_file, print_read_data=True, add_xmax=True)
