import numpy as np
from dstparser import parse_dst_file
from time import time
from dstparser.cli.cli import parse_config
import sys
import itertools
import awkward as ak


def check_parser(dst_file, print_read_data=False):
    """
    Tests the dstparser with various parameter combinations.
    """
    if len(sys.argv) > 1:
        config = parse_config(sys.argv[1])
    else:
        config = None

    # Create all combinations of boolean parameters
    param_options = [True, False]
    param_combinations = list(
        itertools.product(
            param_options, param_options, param_options, param_options
        )
    )

    for use_grid_model, avg_traces, add_shower_params, add_standard_recon in param_combinations:
        print("-" * 80)
        print(
            f"Testing with: use_grid_model={use_grid_model}, avg_traces={avg_traces}, "
            f"add_shower_params={add_shower_params}, add_standard_recon={add_standard_recon}"
        )

        start = time()
        data = parse_dst_file(
            dst_file,
            ntile=7,
            xmax_reader=None,
            avg_traces=avg_traces,
            add_shower_params=add_shower_params,
            add_standard_recon=add_standard_recon,
            config=config,
            use_grid_model=use_grid_model,
        )
        end = time()
        print(f"Parse time: {end - start:.3f} sec")

        if data is None:
            print("Parser returned None. Skipping checks for this combination.")
            continue

        if print_read_data:
            print(f"\nConverted arrays:\n---")
            for key, val in data.items():
                if isinstance(val, np.ndarray):
                    print(key, val.shape)
                elif isinstance(val, ak.Array):
                    print(key, f"ak.Array with type {val.type}")
                else:
                    print(key, len(val))

        # --- Assertions ---
        if add_shower_params:
            assert "energy" in data, "energy should be in data"
        else:
            assert "energy" not in data, "energy should not be in data"

        if add_standard_recon:
            assert "std_recon_energy" in data, "std_recon_energy should be in data"
        else:
            assert "std_recon_energy" not in data, "std_recon_energy should not be in data"

        if use_grid_model:
            assert "detector_positions" in data
            if avg_traces:
                assert "arrival_times" in data
                assert "arrival_times_low" not in data
            else:
                assert "arrival_times_low" in data
                assert "arrival_times" not in data
        else:  # Awkward model
            assert "hits_det_id" in data
            if avg_traces:
                assert "hits_time_traces" in data
                assert "hits_time_traces_low" not in data
            else:
                assert "hits_time_traces_low" in data
                assert "hits_time_traces" not in data
        
        print("Checks passed for this combination.")

    if config is not None and data is not None:
        print("-" * 80)
        print("Checking config-based event IDs:")
        print(f'event_id = {data.get("id_event")}')
        print(f'corsika_shower_id = {data.get("id_corsika_shower")}')
        print(f'energy_bin_id = {data.get("id_energy_bin")}')
        print(f'data_set_id = {data.get("id_data_set")}')


if __name__ == "__main__":
    dst_file = (
        "/ceph/work/SATORI/projects/TA-ASIoP/INR_group/cluster82/grisha/tasdmc_SIBYLL_fe/p2/"
        "DAT013520.corsika77420.SIBYLL.tar.gz.spctr1.1945.noCuts.dst.gz"
    )
    check_parser(dst_file, print_read_data=False)
