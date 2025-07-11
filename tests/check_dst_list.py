import numpy as np
from time import time
import sys
import itertools
import awkward as ak

from dstparser.dst_reader import read_dst_file
from dstparser.dst_parsers import parse_dst_string
from dstparser import parse_dst_file
from dstparser.cli.cli import parse_config


def get_ak_shape(t):
    """Recursively builds a shape tuple from an Awkward array type."""
    if isinstance(t, ak.types.ListType):
        # For variable-length dimensions, we represent them as None
        return (None,) + get_ak_shape(t.content)
    elif isinstance(t, ak.types.NumpyType):
        # Base case: we've reached the primitive type, so we stop here
        return ()
    elif isinstance(t, ak.types.RecordType):
        # For record types, we can't give a simple shape, so we indicate its presence
        return ("record",)
    elif t is None:
        return ()
    else:
        # Handle other unknown types gracefully
        return ("unknown",)


def get_concrete_shape(arr):
    """Recursively builds a shape tuple for a concrete array (NumPy or Awkward)."""
    if isinstance(arr, (np.ndarray, ak.Array)):
        # For Awkward Array, this gets the length of the outermost dimension.
        # For NumPy array, it's the first element of its shape.
        if len(arr) == 0:
            return (0,)
        # Recurse on the first element to get the shape of inner dimensions.
        return (len(arr),) + get_concrete_shape(arr[0])
    else:
        # Base case: element is not an array, so we're at the end of the dimensions.
        return ()


def check_dst_list(dst_file, print_read_data=False):
    # Load config if provided
    if len(sys.argv) > 1:
        config = parse_config(sys.argv[1])
    else:
        config = None

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

        # Parse DST into structured data dict
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
        
        final_data = data  # Save last successful parse for final check

        if print_read_data:
            print("\nParsed data keys, types, and shapes:")
            for key, val in data.items():
                if isinstance(val, np.ndarray):
                    print(f"  - {key}: numpy array, shape {val.shape}")
                elif isinstance(val, ak.Array):
                    # Prepend the number of events (the first dimension) to the shape tuple
                    shape_tuple = (len(val),) + get_ak_shape(val.type.content)
                    print(f"  - {key}: awkward array, shape {shape_tuple}")
                    # Print shape of first few elements for inspection
                    print(f"    Sub-array shapes for first 3 events:")
                    for idx, sub_array in enumerate(val[:3]):  # Check first 3 events
                        sub_shape = get_concrete_shape(sub_array)
                        print(f"    - {key}[{idx}]: shape {sub_shape}")
                elif isinstance(val, list):
                    print(f"  - {key}: list, length {len(val)}")
                else:
                    print(f"  - {key}: type {type(val)}")
        
        print("... combination processed.")


    # Print config-based IDs if available from the last run
    if config is not None and final_data is not None:
        print("-" * 80)
        print("Config-based IDs from last successful run:")
        print(f'id_event = {final_data.get("id_event", "N/A")}')
        print(f'id_corsika_shower = {final_data.get("id_corsika_shower", "N/A")}')
        print(f'id_energy_bin = {final_data.get("id_energy_bin", "N/A")}')
        print(f'id_data_set = {final_data.get("id_data_set", "N/A")}')


if __name__ == "__main__":
    dst_file = (
        "/ceph/work/SATORI/projects/TA-ASIoP/INR_group/cluster82/grisha/"
        "tasdmc_SIBYLL_fe/p2/"
        "DAT013520.corsika77420.SIBYLL.tar.gz.spctr1.1945.noCuts.dst.gz"
    )
    check_dst_list(dst_file, print_read_data=True)
