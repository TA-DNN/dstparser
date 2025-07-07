import numpy as np
from time import time
import sys

from dstparser.dst_reader import read_dst_file
from dstparser.dst_parsers import parse_dst_string
from dstparser import parse_dst_file
from dstparser.cli.cli import parse_config
import awkward as ak


def test_parser(dst_file, print_read_data=False):
    # Load config if provided
    config = parse_config(sys.argv[1]) if len(sys.argv) > 1 else None

    # Parse DST into an Awkward RecordArray
    start = time()
    rec = parse_dst_file(
        dst_file,
        xmax_reader=None,
        avg_traces=False,
        add_shower_params=True,
        add_standard_recon=True,
        config=config,
    )
    end = time()
    print(f"Parse time: {end - start:.3f} sec")

    # Bail out if nothing was parsed
    if rec is None or len(rec) == 0:
        print("No data parsed from DST file.")
        return

    if print_read_data:
        # Inspect raw DST sections
        dst_string = read_dst_file(dst_file)
        dst_lists = parse_dst_string(dst_string)
        section_names = ['event_list', 'sdmeta_list', 'sdwaveform_list', 'badsdinfo_list']
        print("\nRaw DST sections and shapes:")
        if dst_lists is None:
            print("No DST sections (empty file).")
        else:
            for name, section in zip(section_names, dst_lists):
                if isinstance(section, np.ndarray):
                    print(f"{name}: array shape {section.shape}")
                elif isinstance(section, list):
                    print(f"{name}: list length {len(section)}")
                    for idx, arr in enumerate(section[:5]):
                        if isinstance(arr, np.ndarray):
                            print(f"  {name}[{idx}]: array shape {arr.shape}")
                        else:
                            print(f"  {name}[{idx}]: type {type(arr)}")

        # Inspect the Awkward record
        print("\nAwkward record fields:", ak.fields(rec))
        print("First event record:", rec[0])

    # Print config‐based IDs if available
    if config is not None:
        def get_field(name):
            return rec[name][0] if name in ak.fields(rec) else "N/A"

        print(f"id_event           = {get_field('id_event')}")
        print(f"id_corsika_shower  = {get_field('id_corsika_shower')}")
        print(f"id_energy_bin      = {get_field('id_energy_bin')}")
        print(f"id_data_set        = {get_field('id_data_set')}")


if __name__ == "__main__":
    dst_file = (
        "/ceph/work/SATORI/projects/TA-ASIoP/INR_group/cluster82/grisha/"
        "tasdmc_SIBYLL_fe/p2/"
        "DAT013520.corsika77420.SIBYLL.tar.gz.spctr1.1945.noCuts.dst.gz"
    )
    test_parser(dst_file, print_read_data=True)
