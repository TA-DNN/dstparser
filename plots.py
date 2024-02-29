import matplotlib.pyplot as plt
import numpy as np


def waveform_plot(data, event_index):
    ## plot wavefroms (average of lower and upper layers) of ntille x ntile SDs
    ntime_trace = data["time_traces"].shape[3]
    ntile = data["time_traces"].shape[1]
    time = np.arange(0, ntime_trace * 20, 20) # ns
    offset_step = 5
    offset = 0
    colormap = plt.get_cmap("rainbow")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    final_time = max([value for row in data["arrival_times"][event_index] for value in row])
    plot_list = []
    for i in range(ntile):
        for j in range(ntile):
            if np.any(data["time_traces"][event_index][i][j] != 0):
                plot_list.append([data["arrival_times"][event_index][i][j],
                                  data["time_traces"][event_index][i][j],
                                  colormap(
                                      (data["arrival_times"][event_index][i][j] - 0) / (final_time - 0)
                                  ),
                                  "[%d, %d]"%(i, j)
                              ])
                offset += offset_step
    plot_list = sorted(plot_list,
                       key=lambda x: x[0])
    for i in range(len(plot_list)):
        ax.step(
            plot_list[i][0] + time,
            plot_list[i][1] + offset_step * i,
            color = plot_list[i][2],
            where = "mid"
        )
        ax.text(
            plot_list[i][0] + ntime_trace * 20,
            offset_step * i,
            plot_list[i][3],
            color = plot_list[i][2]
        )
    ax.set_xlabel("time [ns]")
    ax.set_ylabel("VEM / time bin")
    ax.set_title(
        r"$E_{gen}$ = %.2f EeV, $\theta_{gen}$ = %.2f deg, $\phi_{gen}$ = %.2f deg"
        %(
            data["energy"][event_index],
            np.arccos(data["shower_axis"][event_index][2]) * 180 / np.pi,
            np.arctan2(data["shower_axis"][event_index][1],
                       data["shower_axis"][event_index][0]) * 180 / np.pi 
            if np.arctan2(data["shower_axis"][event_index][1],
                       data["shower_axis"][event_index][0]) > 0
            else (np.arctan2(data["shower_axis"][event_index][1],
                       data["shower_axis"][event_index][0]) * 180 / np.pi) + 360
        )
    )
    return fig, ax

def tile_plot(data, event_index):
    ## plot ntile x ntile grids in absolute value
    ##      with air shower core position and direction
    ##  - SD status by marker (square for good SD, x for bad SD)
    ##  - signal size by marker size
    ##  - arrival time by color
    ntile = data["time_traces"].shape[1]
    colormap = plt.get_cmap("rainbow")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    final_time = max([value for row in data["arrival_times"][event_index] for value in row])
    for i in range(ntile):
        for j in range(ntile):
            ax.scatter(
                data["detector_positions"][event_index][i][j][0],
                data["detector_positions"][event_index][i][j][1],
                edgecolors = (
                    "black"
                    if np.any(data["time_traces"][event_index][i][j] != 0)
                    else "white"
                ),
                color = (
                    colormap(
                        (data["arrival_times"][event_index][i][j] - 0) / (final_time - 0)
                    )
                    if np.any(data["time_traces"][event_index][i][j] != 0)
                    else "black"
                ),
                cmap = "rainbow",
                vmin = 0,
                vmax = final_time,
                marker = "s" if np.any(data["detector_states"][event_index][i][j] != 0) else "x",
                s = 100 * np.log10(sum(data["time_traces"][event_index][i][j])) + 1 if np.any(data["time_traces"][event_index][i][j] != 0) else 50
            )
    arrow_length = 1000 # meter
    ax.quiver(
        data["shower_core"][event_index][0] + arrow_length / 2 * np.cos(np.arctan2(data["shower_axis"][event_index][1], data["shower_axis"][event_index][0])),
        data["shower_core"][event_index][1] + arrow_length / 2 * np.sin(np.arctan2(data["shower_axis"][event_index][1], data["shower_axis"][event_index][0])),
        arrow_length * np.cos(np.pi + np.arctan2(data["shower_axis"][event_index][1], data["shower_axis"][event_index][0])),
        arrow_length * np.sin(np.pi + np.arctan2(data["shower_axis"][event_index][1], data["shower_axis"][event_index][0])),
        angles = "xy",
        scale_units = "xy",
        scale = 1
    )
    ax.scatter(
        data["shower_core"][event_index][0],
        data["shower_core"][event_index][1],
        marker="*",
        color="white",
        edgecolors="black",
        s = 100
    )
    scat = ax.scatter([], [], c = [], cmap = "rainbow", vmin = 0, vmax = final_time)
    cbar = plt.colorbar(scat, ax=ax)
    cbar.set_label(r"relative time [ns]", labelpad=20, rotation=270, fontsize=12)
    ax.set_xlabel("CLF-X [m]")
    ax.set_ylabel("CLF-Y [m]")
    ax.set_title(
        r"$E_{gen}$ = %.2f EeV, $\theta_{gen}$ = %.2f deg, $\phi_{gen}$ = %.2f deg"
        %(
            data["energy"][event_index],
            np.arccos(data["shower_axis"][event_index][2]) * 180 / np.pi,
            np.arctan2(data["shower_axis"][event_index][1],
                       data["shower_axis"][event_index][0]) * 180 / np.pi 
            if np.arctan2(data["shower_axis"][event_index][1],
                       data["shower_axis"][event_index][0]) > 0
            else (np.arctan2(data["shower_axis"][event_index][1],
                       data["shower_axis"][event_index][0]) * 180 / np.pi) + 360
        )
    )
    plt.gca().set_aspect("equal", adjustable="box")
    return fig, ax

def tile_index_plot(data, event_index):
    ## plot ntile x ntile grids
    ##  - SD status by marker (square for good SD, x for bad SD)
    ##  - signal size by marker size
    ##  - arrival time by color
    ntile = data["time_traces"].shape[1]
    colormap = plt.get_cmap("rainbow")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    final_time = max([value for row in data["arrival_times"][event_index] for value in row])
    for i in range(ntile):
        for j in range(ntile):
            ax.scatter(
                i,
                j,
                edgecolors = (
                    "black"
                    if np.any(data["time_traces"][event_index][i][j] != 0)
                    else "white"
                ),
                color = (
                    colormap(
                        (data["arrival_times"][event_index][i][j] - 0) / (final_time - 0)
                    )
                    if np.any(data["time_traces"][event_index][i][j] != 0)
                    else "black"
                ),
                cmap = "rainbow",
                vmin = 0,
                vmax = final_time,
                marker = "s" if np.any(data["detector_states"][event_index][i][j] != 0) else "x",
                s = 100 * np.log10(sum(data["time_traces"][event_index][i][j])) + 1 if np.any(data["time_traces"][event_index][i][j] != 0) else 50
            )
    scat = ax.scatter([], [], c = [], cmap = "rainbow", vmin = 0, vmax = final_time)
    cbar = plt.colorbar(scat, ax=ax)
    cbar.set_label(r"relative time [ns]", labelpad=20, rotation=270, fontsize=12)
    ax.set_xlabel("tile index X")
    ax.set_ylabel("tile index Y")
    ax.set_title(
        r"$E_{gen}$ = %.2f EeV, $\theta_{gen}$ = %.2f deg, $\phi_{gen}$ = %.2f deg"
        %(
            data["energy"][event_index],
            np.arccos(data["shower_axis"][event_index][2]) * 180 / np.pi,
            np.arctan2(data["shower_axis"][event_index][1],
                       data["shower_axis"][event_index][0]) * 180 / np.pi 
            if np.arctan2(data["shower_axis"][event_index][1],
                       data["shower_axis"][event_index][0]) > 0
            else (np.arctan2(data["shower_axis"][event_index][1],
                       data["shower_axis"][event_index][0]) * 180 / np.pi) + 360
        )
    )
    plt.gca().set_aspect("equal", adjustable="box")
    return fig, ax

###
event_index = 1
ifPlot = False

## plot
if ifPlot:
    from matplotlib.colors import LogNorm

    ## 1) Waveform
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    time128bins = np.arange(0, 128, 1)
    offsetValue = 300
    offset = -1 * offsetValue
    prevSD = 0
    for wf in np.array(sdwaveform_list[event_index]):
        alpha = 1
        if (
            next(item[1] for item in sdmeta_list[event_index] if item[0] == wf[0]) <= 2
        ):  ## plot only space-time cluster SDs
            # alpha = 0.1
            continue
        if wf[0] != prevSD:
            offset += offsetValue
        timeStart = wf[1] / wf[2] * 1.0 * (10**9) / 20  # 1 bin = 20 ns
        ax.plot(
            timeStart + time128bins, wf[3 : 3 + 128] + offset, color="red", alpha=alpha
        )
        ax.plot(
            timeStart + time128bins, wf[3 + 128 :] + offset, color="blue", alpha=alpha
        )
        ax.text(timeStart, offset, wf[0])
        prevSD = wf[0]
    ax.set_xlabel("time bin [20 ns]")
    ax.set_ylabel("FADC count")
    ax.set_title(
        r"$E_{gen}$ = %.2f EeV, $\theta_{gen}$ = %.2f deg, $\phi_{gen}$ = %.2f deg"
        % (
            event_list[event_index][1],
            event_list[event_index][2] * 180 / np.pi,
            event_list[event_index][3] * 180 / np.pi,
        )
    )
    fig.savefig("Waveform_event_index_%d.png" % (event_index))

    ## 2) Foot Print
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sd_show = []
    for j in range(len(sdmeta_list[event_index])):
        if sdmeta_list[event_index][j][1] > 2:
            sd_show.append(
                [
                    str(int(sdmeta_list[event_index][j][0])).zfill(4),
                    rufptn_xxyy2sds(int(sdmeta_list[event_index][j][0]))[1] / 100,
                    rufptn_xxyy2sds(int(sdmeta_list[event_index][j][0]))[2] / 100,
                    (sdmeta_list[event_index][j][4] + sdmeta_list[event_index][j][5])
                    / 2,
                    (sdmeta_list[event_index][j][2] + sdmeta_list[event_index][j][3])
                    / 2,
                ]
            )
    sd_show = np.array(sd_show, dtype=float)
    print(sd_show)
    scat = ax.scatter(
        sd_show[:, 1],
        sd_show[:, 2],
        s=np.log10(sd_show[:, 3]) * 100 + 1,
        c=sd_show[:, 4] * 4 * 1000 - min(sd_show[:, 4] * 4 * 1000),
        vmin=0,
        vmax=10**4,
        cmap="rainbow",
    )
    # vmin = min(sd_show[:,4]),
    # vmax = max(sd_show[:,4]))
    cbar = plt.colorbar(scat, ax=ax)
    cbar.set_label(r"relative time [ns]", labelpad=20, rotation=270, fontsize=12)
    for i in range(tasd_clf.tasdmc_clf.shape[0]):
        ax.scatter(
            tasd_clf.tasdmc_clf[i, 1] / 100,
            tasd_clf.tasdmc_clf[i, 2] / 100,
            marker="s",
            c="white",
            s=3,
            edgecolors="black",
        )
    plt.gca().set_aspect("equal", adjustable="box")
    ax.scatter(
        event_list[event_index][4] / 100,
        event_list[event_index][5] / 100,
        marker="*",
        color="white",
        edgecolors="black",
    )
    ax.set_xlabel("CLF-x [m]")
    ax.set_ylabel("CLF-y [m]")
    fig.savefig("FootPrint_event_index_%d.png" % (event_index))

    ## 3) nTile x nTile SD states
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(nTile):
        for j in range(nTile):
            ax.scatter(
                i,
                j,
                edgecolors=(
                    "black" if np.any(time_traces[event_index][i][j] != 0) else "white"
                ),
                c=(
                    arrival_times[event_index][i][j]
                    if np.any(time_traces[event_index][i][j] != 0)
                    else "black"
                ),
                cmap="rainbow",
                vmin=0,
                vmax=10**4,
                marker="s" if detector_states[event_index][i][j] else "x",
                s=100 if np.any(time_traces[event_index][i][j] != 0) else 50,
            )
    plt.gca().set_aspect("equal", adjustable="box")
    # scat = ax.scatter([],[],
    #           c = [],
    #           cmap = "rainbow",
    #          vmin = 0,
    #          vmax = 10**4)
    cbar = plt.colorbar(scat, ax=ax)
    cbar.set_label(r"relative time [ns]", labelpad=20, rotation=270, fontsize=12)
    ax.set_title(r"%d $\times$ %d SD grids for DNN" % (nTile, nTile))
    ax.set_xlabel("tile index X")
    ax.set_ylabel("tile index Y")
    fig.savefig("Tilesfor_DNN_event_index_%d.png" % (event_index))

    ## 4) nTile x nTile time trace
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colormap = plt.get_cmap("rainbow")
    step = 10
    for i in range(nTile):
        for j in range(nTile):
            ax.step(
                np.arange(nTimeTrace),
                time_traces[event_index][i][j] + step * (i * nTile + j),
                where="mid",
                # c = arrival_times[event_index][i][j],
                color=(
                    colormap((arrival_times[event_index][i][j] - 0) / (10**4 - 0))
                    if np.any(time_traces[event_index][i][j] != 0)
                    else "black"
                ),
            )
    ax.set_title(r"%d $\times$ %d time traces" % (nTile, nTile))
    ax.set_xlabel("time trace [time bin (20 ns/bin)]")
    ax.set_ylabel("VEM / time bin (float32)")
    fig.savefig("TimeTrace_forDNN_event_index_%d.png" % (event_index))
    ##
    plt.show()
