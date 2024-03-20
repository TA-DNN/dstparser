import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from pathlib import Path
from srecog.utils.hdf5_utils import dict_from_file, array_from_file
from mpl_toolkits.axes_grid1 import make_axes_locatable


def order_time_traces(arrival_times, time_traces):
    list_to_sort = []
    for i in range(time_traces.shape[0]):
        for j in range(time_traces.shape[1]):
            if arrival_times[i, j] > 0:
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


def event_title(data_file, event_idx):

    data = dict_from_file(
        data_file,
        ["energy", "shower_axis", "xmax"],
        indices=slice(event_idx, event_idx + 1),
    )
    for key, value in data.items():
        data[key] = value.squeeze()

    energy = data["energy"]
    shower_axis = data["shower_axis"]
    xmax = data["xmax"]

    z_axis = np.array([0, 0, 1])
    cos_theta = np.dot(shower_axis, z_axis) / np.linalg.norm(shower_axis)
    zenith = np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi

    title = (
        r"$E$ = %.2f EeV, $\theta$ = %.2f deg, $X_{max}$ = %.2f $g/cm^2$"
        % (
            energy,
            zenith,
            xmax,
        )
    )
    file_name = Path(data_file).stem
    file_name = f"{file_name}_{event_idx:03}"
    return title, file_name


def read_time_traces(data_file, event_idx=0):
    data = dict_from_file(
        data_file,
        ["time_traces", "detector_states", "arrival_times"],
        indices=slice(event_idx, event_idx + 1),
    )

    for key, value in data.items():
        data[key] = value.squeeze()

    data["time_traces"] = np.log10(data["time_traces"] + 1)
    return data


def extend_time_traces(arrival_times, time_traces):

    # arrival times to bins
    nsec_per_bin = 20  # ns
    arrival_times = np.ceil(arrival_times / nsec_per_bin).astype(np.int32)

    # Create the long time traces
    trace_length = time_traces.shape[2]
    time_traces_long = np.zeros((9, 9, arrival_times.max() + trace_length))

    # Shift time traces according arrival_times if arrival_times > 0
    for i in range(arrival_times.shape[0]):
        for j in range(arrival_times.shape[1]):
            arrival_pos = arrival_times[i, j]
            if arrival_pos > 0:
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
    sig_detectors = np.stack(np.where(arrival_times > 0)).transpose()

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
    
    for i in range(time_traces.shape[2]):
        im = ax.imshow(
            time_traces[:, :, i],
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
            colormap = plt.get_cmap("turbo")
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(label)  # Set color bar label
            for row, col in bad_detectors:
                ax.plot(col, row, "rx", markersize=16)
                   

            ttime, ttraces = order_time_traces(arrival_times, time_traces)
            
            max_arr_time = np.max(arrival_times)
            arr_time_color = arrival_times/max_arr_time
            for row, col in sig_detectors:
                ax.plot(col, row, ".", 
                        markersize=15,
                        color = colormap(arr_time_color[row, col]))

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
):
    # cmap = "viridis"
    # cmap = "Greys"
    data = read_time_traces(data_file, event_idx)
    arrival_times = data["arrival_times"]
    time_traces = data["time_traces"]
    detector_states = data["detector_states"]

    title, file_name = event_title(data_file, event_idx)

    if only_traces:
        tile_signal(
            time_traces[:, :, time_slice],
            arrival_times,
            detector_states=detector_states,
            out_file=f"{file_name}_trace",
            cmap=cmap,
            scale=scale,
            title=title,
            interval=interval,
        )
    else:
        time_traces_long = extend_time_traces(arrival_times, time_traces)
        tile_signal(
            time_traces_long[:, :, time_slice],
            arrival_times,
            detector_states=detector_states,
            out_file=f"{file_name}",
            cmap=cmap,
            scale=scale,
            title=title,
            interval=interval,
        )
