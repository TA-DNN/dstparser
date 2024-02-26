import matplotlib.pyplot as plt
import numpy as np

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
    offset = 0
    prevSD = -1 * offsetValue
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