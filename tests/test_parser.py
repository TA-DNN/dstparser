import numpy as np
from dstparser import parse_dst_file
from time import time
from dstparser.cli.cli import parse_config
import sys


def test_parser(dst_file, print_read_data=False):
    # Load configuration if provided
    if len(sys.argv) > 1:
        config = parse_config(sys.argv[1])
    else:
        config = None

    start = time()
    # Parse DST file: returns a pandas DataFrame in the new adapter
    df = parse_dst_file(
        dst_file,
        xmax_reader=None,
        avg_traces=False,
        add_shower_params=True,
        add_standard_recon=True,
        config=config
    )
    elapsed = time() - start
    print(f"Parsing took {elapsed:.2f} seconds, loaded {len(df)} events.")

    if print_read_data:
        # Show the first few rows of the DataFrame
        print(df.head())

    if config is not None and len(df):
        # Print key metadata for the first event
        first = df.iloc[0]
        print(f"event_id           = {first['id_event']}")
        print(f"corsika_shower_id  = {first['id_corsika_shower']}")
        print(f"energy_bin_id      = {first['id_energy_bin']}")
        print(f"data_set_id        = {first['id_data_set']}")


if __name__ == "__main__":
    dst_file = (
        "/ceph/work/SATORI/projects/TA-ASIoP/INR_group/cluster82/grisha/tasdmc_SIBYLL_fe/p2/"
        "DAT013520.corsika77420.SIBYLL.tar.gz.spctr1.1945.noCuts.dst.gz"
    )
    test_parser(dst_file, print_read_data=False)
