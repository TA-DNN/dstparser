import numpy as np
from dstparser import parse_dst_file, parse_dst_file_tax4, parse_dst_file_vlen
from time import time
from dstparser.cli.cli import parse_config
import sys


def test_parser(dst_file, print_read_data=False, parse_type="vlen_ta", add_xmax=False):

    if len(sys.argv) > 1:
        config = parse_config(sys.argv[1])
    else:
        config = None
    start = time()

    # The only function that are required to parsing
    # is parse_dst_file
    # paths to the directories with data is dstparser.paths module
    # data = parse_dst_file(dst_file, up_low_traces=True)

    parse_types = {
        "grid_ta": parse_dst_file,
        "grid_ta_x4": parse_dst_file_tax4,
        "vlen_ta": parse_dst_file_vlen,
    }

    convert_func = parse_types[parse_type]

    # if add_xmax:
    #     from dstparser.xmax_reader import XmaxReader

    #     xmax_dir = "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04proton/080417_160603/Em1_bsdinfo"
    #     xmax_reader = XmaxReader(xmax_dir, "**/DAT*_xmax.txt", "QGSJetII-04")
    # else:
    #     xmax_reader = None

    if add_xmax:
        from dstparser.xmax_reader.xmax_reader import create_xmax_reader

        xmax_dbs = {
            "eposlhc": "/ceph/work/SATORI/projects/TA-ASIoP/dnn_training_data/2026/02/xmax_db/eposlhc_xmax_db.h5",
            "qgsjetii04": "/ceph/work/SATORI/projects/TA-ASIoP/dnn_training_data/2026/02/xmax_db/qgsii04_xmax_db.h5",
        }
        xmax_reader = create_xmax_reader(xmax_dbs)
    else:
        xmax_reader = None

    if parse_type == "vlen_ta":
        data = convert_func(
            dst_file,
            xmax_reader=xmax_reader,
            add_shower_params=True,
            add_standard_recon=True,
            config=config,
        )
    else:
        data = convert_func(
            dst_file,
            ntile=7,
            xmax_reader=xmax_reader,
            avg_traces=False,
            add_shower_params=True,
            add_standard_recon=True,
            config=config,
        )

    end = time()

    if print_read_data:
        print(f"\nConverted arrays:\n---")
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                print(key, val.shape)
            else:
                print(key, len(val), val)

    # print(f'energy = {data["energy"]}')
    # print(f'xmax = {data["xmax"]}')
    # print(f'time_traces_low = {data["time_traces_low"]}')
    # print(f'event id = {data["event_id"]}')
    # print(f'corsika shower id = {data["corsika_shower_id"]}')
    # print(f'energy bin id = {data["energy_bin_id"]}')
    # print(f"detector_positions_id = {data['detector_positions_id'][50]}")
    # print(f"detector_positions_abs = {data['detector_positions_abs'][50][:, :, 0]}")
    # print(f"detector_positions = {data['detector_positions'][50][:, :, 0].astype(np.float16)}")
    # print(f"detector_positions_id = {data['detector_positions_abs'][50].shape}")
    # print("xmax=", data["xmax"])
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
    # dst_file = "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04proton/080417_160603/Em1_bsdinfo/DAT051419_gea.rufldf.dst.gz"
    # dst_file = "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/tax4/qgsii04proton/north/221101to240124/DAT010019_gea.rufldf.dst.gz"
    # dst_file = "/ceph/work/SATORI/projects/TA-ASIoP/INR_group/cluster82/grisha/tasdmc_EPOS_p/p2/DAT000623.corsika77420.EPOS.tar.gz.spctr1.1745.noCuts.dst.gz"
    dst_file = "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii04nitrogen/160604_240422/Em1_bsdinfo/XXXX22/DAT000022_gea.rufldf.dst.gz"
    # !If you want to use TAx4 format, set use_ta_x4=True
    # !If you want to use TA format, set use_ta_x4=False
    # !By default, use_ta_x4=False
    test_parser(dst_file, print_read_data=True, parse_type="vlen_ta", add_xmax=True)
