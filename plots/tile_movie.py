import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from pathlib import Path
from srecog.utils.hdf5_utils import dict_from_file, read_hdf5_metadata, arrays_from_file
from mpl_toolkits.axes_grid1 import make_axes_locatable


def order_time_traces(arrival_times, time_traces):
    list_to_sort = []
    for i in range(time_traces.shape[0]):
        for j in range(time_traces.shape[1]):
            if np.sum(time_traces[i, j]) > 0:
                list_to_sort.append([arrival_times[i, j], time_traces[i, j]])
    ttraces = sorted(list_to_sort, key=lambda x: x[0])

    micro_sec = 1e-3
    bin_time = 20  # ns
    ttime = np.arange(0, len(ttraces[0][1]) * bin_time, bin_time) * micro_sec

    return ttime, ttraces


def plot_time_series(arrival_times, time_traces_long):
    lst = []
    for i in range(time_traces_long.shape[0]):
        for j in range(time_traces_long.shape[1]):
            if arrival_times[i, j] > 0:
                lst.append([arrival_times[i, j], time_traces_long[i, j]])

    lst = sorted(lst, key=lambda x: x[0])

    colormap = plt.get_cmap("turbo")
    # colormap = plt.get_cmap("rainbow")
    cc = 0

    xl = np.arange(0, len(lst[0][1]) * 20, 20) / 1e3

    fig, ax = plt.subplots(figsize=(10, 3))
    for lll in lst:
        ttplot = lll[1]
        ttplot = ttplot + 0.05 * cc
        mask = lll[1] > 0
        # ttplot = np.exp(ttplot - 1) + 0.05 * cc
        plt.step(
            xl[mask],
            ttplot[mask],
            #  color="black",
            color=colormap(cc / len(lst)),
            linewidth=0.5,
        )
        cc += 1

    plt.xlabel(r"$\mu s$")


def read_events(data_file):
    all_arrays = [
        "arrival_times",
        "detector_positions",
        "detector_states",
        "energy",
        "mass_number",
        "metadata",
        "shower_axis",
        "shower_core",
        "time_traces",
        "time_traces_low",
        "time_traces_up",
        "xmax",
    ]
    return dict_from_file(data_file, all_arrays)


def event_title(data):
    z_axis = np.array([0, 0, 1])
    cos_theta = np.dot(data["shower_axis"], z_axis) / np.linalg.norm(
        data["shower_axis"]
    )
    zenith = np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi

    title = r"$E$ = %.2f EeV, $\theta$ = %.2f deg, $X_{max}$ = %.2f $g/cm^2$" % (
        data["energy"],
        zenith,
        data["xmax"],
    )

    return title


def out_file_ta(data_file, event_idx, out_dir=None):
    file_name = Path(data_file).stem
    dir_name = Path(data_file).parent.name
    if out_dir is not None:
        out_dir = Path(out_dir) / dir_name / file_name
        out_dir.mkdir(parents=True, exist_ok=True)
        file_name = str(out_dir / f"{file_name}_{event_idx:03}")
    else:
        file_name = f"{file_name}_{event_idx:03}"

    return file_name


def read_data_ta(data_file, event_idx):
    data = dict_from_file(
        data_file,
        [
            "energy",
            "shower_axis",
            "xmax",
            "time_traces",
            "detector_states",
            "arrival_times",
            "detector_positions",
            "shower_core",
            "shower_axis",
        ],
        indices=slice(event_idx, event_idx + 1),
    )

    for key, value in data.items():
        data[key] = value.squeeze()

    data["time_traces"] = np.log10(data["time_traces"] + 1)
    return data


def read_event_ta(data_file, event_idx, out_dir=None):
    data = read_data_ta(data_file, event_idx)
    out_file = out_file_ta(data_file, event_idx, out_dir)
    return data, out_file


def read_event_toy(data_file, event_idx, out_dir=None):
    """convert from toy simulation format to new (TA) format"""
    array_names = []
    meta_data = read_hdf5_metadata(data_file)
    for key in meta_data:
        if key not in ["file", "settings", "detector"]:
            array_names.append(key)

    idxs = slice(event_idx, event_idx + 1)
    original_data = dict_from_file(data_file, array_names, indices=idxs)

    data = dict()
    data["arrival_times"] = original_data["detector_readings"][:, :, :, 0]
    data["time_traces"] = original_data["time_traces"][:, :, :, :]
    data["energy"] = original_data["energy"]
    data["xmax"] = original_data["xmax"]
    data["shower_axis"] = original_data["showeraxis"]
    data["shower_core"] = original_data["showercore"]/(4*1200)
    ntile = data["arrival_times"].shape[1]
    data["detector_positions"] = arrays_from_file(data_file, "detector").reshape(
        ntile, ntile, 3
    ) / (4 * 1200)
    data["detector_states"] = np.ones((9, 9), dtype=bool)

    for key, value in data.items():
        data[key] = value.squeeze()
    
    print(f'Arrival times in toy DNN {data["arrival_times"]}')
    print(f'shower_axis {data["shower_axis"]}')
    print(f'shower_core {data["shower_core"]}')
    
    arr = data["arrival_times"]
    arr[arr == 0] = np.nan
    arr = arr - np.nanmin(arr)
    arr = arr/np.nanmax(arr)
    arr[np.isnan(arr)] = 0
    data["arrival_times"] = arr*5*1e3
    print(data["arrival_times"])    

    file_name = "ttrace"
    dir_name = Path(data_file).stem
    if out_dir is not None:
        out_dir = Path(out_dir) / dir_name / file_name
        out_dir.mkdir(parents=True, exist_ok=True)
        file_name = str(out_dir / f"{file_name}_{event_idx:07}")
    else:
        file_name = f"{file_name}_{event_idx:03}"

    return data, file_name


def extend_time_traces(arrival_times, time_traces):

    # arrival times to bins
    nsec_per_bin = 20  # ns
    arrival_times = np.ceil(arrival_times / nsec_per_bin).astype(np.int32)
    
    print(arrival_times)

    # Create the long time traces
    trace_length = time_traces.shape[2]
    time_traces_long = np.zeros((9, 9, arrival_times.max() + trace_length))

    # Shift time traces according arrival_times if arrival_times > 0
    for i in range(arrival_times.shape[0]):
        for j in range(arrival_times.shape[1]):
            arrival_pos = arrival_times[i, j]
            if np.sum(time_traces[i, j]) > 0:
                time_traces_long[i, j, arrival_pos : trace_length + arrival_pos] = (
                    time_traces[i, j]
                )
    return time_traces_long


def min_max_vals(time_traces):
    # Find the overall min and max for consistent color bar scaling
    min_val = np.min(time_traces)
    max_val = np.max(time_traces)

    # Ensure that vmin is positive (for logarithmic scale)
    if min_val <= 0:
        if len(time_traces[time_traces > 0]) > 0:
            min_val = np.min(time_traces[time_traces > 0])

    # Ensure that vmax is greater than vmin
    if max_val <= min_val:
        max_val = min_val + 1

    return min_val, max_val


def tile_signal(
    time_traces,
    arrival_times,
    detector_states,
    detector_positions,
    shower_core,
    shower_axis,
    out_file,
    cmap="Blues",
    scale="Linear",
    title="",
    interval=50,
):

    fig, ax = plt.subplots(figsize=(5.5, 6))
    divider = make_axes_locatable(ax)
    ax1 = divider.append_axes("bottom", size=0.8, pad=0.3)
    cax = divider.append_axes("right", size=0.15, pad=0.1)

    # Indicies of bad detectors
    bad_detectors = np.stack(np.where(detector_states == False)).transpose()
    sig_detectors = np.stack(np.where(np.sum(time_traces, axis=-1) > 0)).transpose()

    images = []
    time_per_frame = 20 * 1e-3  # milliseconds per frame
    min_val, max_val = min_max_vals(time_traces)
    if scale == "Log":
        norm = colors.LogNorm(vmin=min_val, vmax=max_val)
        label = "Log Scale"
    else:
        norm = colors.Normalize(vmin=min_val, vmax=max_val)
        label = "Linear Scale"

    # Labels for x, y axis
    ntile = time_traces.shape[0]
    ax.set_xticks(np.arange(ntile))
    # Start enumeration from 1
    ax.set_xticklabels(np.arange(1, ntile + 1))

    ax.set_yticks(np.arange(ntile))
    ax.set_yticklabels(np.arange(1, ntile + 1))

    # Swap axis, because imshow(ny, nx)
    time_traces_yx = time_traces.swapaxes(0, 1)
    arrival_times_yx = arrival_times.swapaxes(0, 1)

    print(f"Len = {time_traces.shape[2]}")
    
    for i in range(time_traces.shape[2]):
        im = ax.imshow(
            time_traces_yx[:, :, i],
            cmap=cmap,
            norm=norm,
            animated=True,
            origin="lower",
        )
        time_count = ax.text(
            0.5,
            1.01,
            f"{title}\nTime {i*time_per_frame:.2f}" + r" $\mu s$",
            size=plt.rcParams["axes.titlesize"],
            ha="center",
            transform=ax.transAxes,
        )
        if i == 0:  # Add color bar only once
            # colormap = plt.get_cmap("winter")
            # colormap1 = plt.get_cmap("jet")
            colormap = plt.get_cmap("rainbow")
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(label)  # Set color bar label
            for row, col in bad_detectors:
                ax.plot(row, col, "rx", markersize=16)

            ttime, ttraces = order_time_traces(arrival_times_yx, time_traces_yx)

            max_arr_time = np.max(arrival_times)
            arr_time_color = arrival_times / max_arr_time
            for row, col in sig_detectors:
                ax.plot(
                    detector_positions[row, col, 0],
                    detector_positions[row, col, 1],
                    ".",
                    markersize=10,
                    color=colormap(arr_time_color[row, col]),
                )

            shower_axis = shower_axis / np.linalg.norm(shower_axis)
            ax.arrow(
                shower_core[0] - shower_axis[0],
                shower_core[1] - shower_axis[1],
                shower_axis[0] * 2,
                shower_axis[1] * 2,
                head_width=0.1,
                fc="black",
                ec="black",
            )

            xperp = -shower_axis[1]
            yperp = shower_axis[0]

            ax.arrow(
                shower_core[0] - xperp * 0.5,
                shower_core[1] - yperp * 0.5,
                xperp,
                yperp,
                fc="black",
                ec="black",
            )

            ax.plot(shower_core[0], shower_core[1], "*", markersize=5, color="red")
            # colormap = plt.get_cmap("rainbow")

            all_plots = []
            ax1.set_xlim(ttime[0], ttime[-1])
            ax1.set_xlabel(r"Time, $\mu s$")
            len_ttraces = len(ttraces)
            for itt, ttrace in enumerate(ttraces):
                arr_time = ttrace[0]
                trace = ttrace[1]
                mask = trace > 0
                trace = trace + 0.05 * itt
                (tt_plot,) = ax1.plot(
                    ttime[mask],
                    trace[mask],
                    color=colormap(arr_time / max_arr_time),
                    linewidth=0.5,
                )
                all_plots.append(tt_plot)

        slider_line = ax1.axvline(ttime[i], color="red")
        images.append([im, slider_line, *all_plots, time_count])

    ani = animation.ArtistAnimation(fig, images, interval=interval, repeat_delay=1000)

    ani.save(f"{out_file}.mp4")
    plt.close()


def tile_signal_movie(
    data_file,
    event_idx,
    cmap="Blues",
    scale="Linear",
    interval=20,
    time_slice=slice(None),
    only_traces=False,
    out_dir=None,
    in_file_format="TA",
):
    # cmap = "viridis"
    # cmap = "Greys"
    if in_file_format == "TA":
        data, out_file = read_event_ta(data_file, event_idx, out_dir)
    elif in_file_format == "TOY":
        data, out_file = read_event_toy(data_file, event_idx, out_dir)
    else:
        raise ValueError(f"{in_file_format} is unknown")

    title = event_title(data)
    arrival_times = data["arrival_times"]
    time_traces = data["time_traces"]
    detector_states = data["detector_states"]
    detector_positions = data["detector_positions"]
    shower_core = data["shower_core"]
    shower_axis = -data["shower_axis"][:2]

    detector_positions = (
        (detector_positions + 1) * (detector_positions.shape[0] - 1) / 2
    )

    shower_core = (shower_core + 1) * (detector_positions.shape[0] - 1) / 2

    if only_traces:
        out_file = f"{out_file}_trace"
    else:
        time_traces = extend_time_traces(arrival_times, time_traces)

    tile_signal(
        time_traces[:, :, time_slice],
        arrival_times,
        detector_states=detector_states,
        detector_positions=detector_positions[:, :, :2],
        shower_core=shower_core,
        shower_axis=shower_axis,
        out_file=out_file,
        cmap=cmap,
        scale=scale,
        title=title,
        interval=interval,
    )
