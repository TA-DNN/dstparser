import numpy as np
from time import time
import sys
import itertools
import awkward as ak
import yaml

from dstparser.dst_reader import read_dst_file
from dstparser.dst_parsers import parse_dst_string
from dstparser import parse_dst_file


def awkward_shape(arr):
    """
    Returns a dict mapping each axis →
      * axis 0: exact length
      * axis ≥1: (min_length, max_length) of that jagged dimension
    """
    layout = ak.to_layout(arr)
    depth = layout.purelist_depth  # number of nested List levels

    summary = {}
    # Axis 0: exact length of the outermost list
    summary[0] = len(arr)

    # Axes 1 through (depth‑1): jagged lengths per level
    for axis in range(1, depth):
        counts = ak.num(arr, axis=axis)  # lengths at this level
        lo, hi = int(ak.min(counts)), int(ak.max(counts))
        summary[axis] = (lo, hi)

    return summary


def check_dst_list(dst_file, config_path, print_read_data=False):
    # Load config if provided
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if print_read_data:
        # Inspect raw DST sections once, as they are independent of parser params
        print("\n" + "=" * 20 + " Raw DST Sections " + "=" * 20)
        dst_string = read_dst_file(dst_file)
        dst_lists = parse_dst_string(dst_string)
        section_names = [
            "event_list",
            "sdmeta_list",
            "sdwaveform_list",
            "badsdinfo_list",
        ]
        if dst_lists is None:
            print("No DST sections found (empty or invalid file).")
        else:
            for name, section in zip(section_names, dst_lists):
                if isinstance(section, np.ndarray):
                    print(f"{name}: array shape {section.shape}")
                elif isinstance(section, list):
                    print(f"{name}: list length {len(section)}")
                    # Print shape of first few elements for inspection
                    for idx, arr in enumerate(section[:3]):  # Check first 3 events
                        if isinstance(arr, np.ndarray):
                            print(f"  {name}[{idx}]: array shape {arr.shape}")
                        else:
                            print(f"  {name}[{idx}]: type {type(arr)}")
        print("=" * 58 + "\n")

    # Create all combinations of boolean parameters for the parser
    param_options = [True, False]
    param_combinations = list(
        itertools.product(param_options, param_options, param_options, param_options)
    )

    final_data = None
    for (
        use_grid_model,
        avg_traces,
        add_shower_params,
        add_standard_recon,
    ) in param_combinations:
        print("-" * 80)
        print(
            f"Testing with: use_grid_model={use_grid_model}, avg_traces={avg_traces}, "
            f"add_shower_params={add_shower_params}, add_standard_recon={add_standard_recon}"
        )

        current_config = config.copy()
        current_config["dst_parser"]["use_grid_model"] = use_grid_model
        current_config["dst_parser"]["avg_traces"] = avg_traces
        current_config["dst_parser"]["add_shower_params"] = add_shower_params
        current_config["dst_parser"]["add_standard_recon"] = add_standard_recon

        data = parse_dst_file(
            dst_file=dst_file,
            ntile=current_config["dst_parser"]["ntile"],
            use_grid_model=use_grid_model,
            avg_traces=avg_traces,
            add_shower_params=add_shower_params,
            add_standard_recon=add_standard_recon,
        )

        if data:
            print("Parsed data shapes:")
            for key, value in data.items():
                if isinstance(value, ak.Array):
                    shape_summary = awkward_shape(value)
                    print(f"  - {key}: {shape_summary}")
                elif isinstance(value, np.ndarray):
                    print(f"  - {key}: {value.shape}")
            final_data = data  # Keep the last valid data for potential further checks
        else:
            print("  -> No data returned from parser.")

    # Print config-based IDs if available from the last run
    if config is not None and final_data is not None:
        print("-" * 80)
        print("Config-based IDs from last successful run:")
        print(f'id_event = {final_data.get("id_event", "N/A")}')
        print(f'id_corsika_shower = {final_data.get("id_corsika_shower", "N/A")}')
        print(f'id_energy_bin = {final_data.get("id_energy_bin", "N/A")}')
        print(f'id_data_set = {final_data.get("id_data_set", "N/A")}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dst_file", help="DST file to process")
    parser.add_argument("config_file", help="YAML config file")
    parser.add_argument("--print", action="store_true", help="Print detailed data info")
    args = parser.parse_args()

    check_dst_list(args.dst_file, args.config_file, print_read_data=args.print)
