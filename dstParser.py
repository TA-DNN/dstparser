## Test init_development 240226

import os
import numpy as np
import sys

os.environ["LD_LIBRARY_PATH"] = (
    os.environ["LD_LIBRARY_PATH"]
    + ":"
    + "/dicos_ui_home/anatoli/groupspace/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM/lib"
)
os.environ["PATH"] = (
    os.environ["PATH"]
    + ":"
    + "/dicos_ui_home/anatoli/groupspace/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM/bin"
)

import subprocess

##
import tasd_clf  ## TASD position (CLF)


def parse_script(dst_file):
    ### Variables
    event_index = 7
    add_label = "DAT000115"
    ifSavePlot = False
    ifPlot = False

    ###
    import time

    start_time = time.time()

    def capture_output(cmd):
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            output, _ = process.communicate()
            return output
        except subprocess.CalledProcessError as e:
            return e.output

    ## read DST file
    output = capture_output("sditerator.run %s" % (dst_file)).strip().split("\n")

    ## make lists
    event_readout = False
    sdmeta_readout = False
    sdwaveform_readout = False
    badsdinfo_readout = False
    event_list_str = []
    sdmeta_list_str = []
    sdwaveform_list_str = []
    badsdinfo_list_str = []
    for il, line in enumerate(output):
        if "EVENT DATA" in line:
            event_readout = True
            sdmeta_readout = False
            sdwaveform_readout = False
            badsdinfo_readout = False
            continue
        elif "SD meta DATA" in line:
            event_readout = False
            sdmeta_readout = True
            sdwaveform_readout = False
            badsdinfo_readout = False
            continue
        elif "SD waveform DATA" in line:
            event_readout = False
            sdmeta_readout = False
            sdwaveform_readout = True
            badsdinfo_readout = False
            continue
        elif "badsdinfo" in line:
            event_readout = False
            sdmeta_readout = False
            sdwaveform_readout = False
            badsdinfo_readout = True
            continue
        if event_readout:
            event_list_str.append(line)
        elif sdmeta_readout:
            sdmeta_list_str.append(line)
        elif sdwaveform_readout:
            sdwaveform_list_str.append(line)
        elif badsdinfo_readout:
            badsdinfo_list_str.append(line)
            

    ##
    ## Make Data set
    ##

    # Meta data
    Interaction_model = "QGSJET-II-03"
    Atmosphere_model = ""
    Emin = ""
    Emax = ""
    Espectrum = "HiRes"
    # DST_file_name = sys.argv[1]
    DST_file_name = dst_file

    def CORSIKAparticleID2mass(corsikaPID):
        if corsikaPID == 14:
            return 1
        else:
            return corsikaPID // 100

    def convert_to_specific_type(columnName, value):
        if columnName == "rusdmc_.parttype":
            return CORSIKAparticleID2mass(int(value))
        elif columnName in [
            "rusdraw_.yymmdd",
            "rusdraw_.hhmmss",
            "rufptn_.nstclust",
            "rusdraw_.nofwf",
            "rufptn_.xxyy",
            "rufptn_.isgood",
        ]:
            return int(value)
        else:
            return float(value)

    # Shower related
    event_list = [[float(c) for c in l.split(" ") if c != ""] for l in event_list_str]

    mass_number = np.array([CORSIKAparticleID2mass(item[0]) for item in event_list], dtype=np.int32)
    energy = np.array([item[1] for item in event_list], dtype=np.float32)
    xmax = np.array([0 for item in event_list])
    shower_axis = np.array(
        [
            [
                np.sin(item[2]) * np.cos(item[3] + np.pi),
                np.sin(item[2]) * np.sin(item[3] + np.pi),
                np.cos(item[2]),
            ]
            for item in event_list
        ],
        dtype=np.float32,
    )
    shower_core = np.array(
        [[item[4]/100, item[5]/100, item[6]/100] for item in event_list], dtype=np.int32
    )

    ## Detection related
    sdmeta_list = [[float(c) for c in l.split(" ") if c != ""] for l in sdmeta_list_str]
    sdmeta_list = [
        [
            [sdmeta_list[i][j + k * 11] for j in range(11)]
            for k in range(len(sdmeta_list[i]) // 11)
        ]
        for i in range(len(sdmeta_list))
    ]
    sdwaveform_list = [
        [int(c) for c in l.split(" ") if c != ""] for l in sdwaveform_list_str
    ]
    sdwaveform_list = [
        [
            [sdwaveform_list[i][j + k * (3 + 128 * 2)] for j in range(3 + 128 * 2)]
            for k in range(len(sdwaveform_list[i]) // (3 + 128 * 2))
        ]
        for i in range(len(sdwaveform_list))
    ]
    badsdinfo_list = [
        [int(c) for c in l.split(" ") if c != ""] for l in badsdinfo_list_str
    ]
    badsdinfo_list = [
        [badsdinfo_list[i][k * 2] for k in range(len(badsdinfo_list[i]) // 2)]
        for i in range(len(badsdinfo_list))
    ]
    badsdinfo_set_list = [{e for e in sublist} for sublist in badsdinfo_list]

    # Put largest-signal SD at the center of nTile x nTile grids
    nTile = 7  # number of SD per one side
    nTimeTrace = 128  # number of time trace of waveform
    #
    arrival_times = np.zeros((len(event_list), nTile, nTile), dtype=np.float32)
    time_traces = np.zeros(
        (len(event_list), nTile, nTile, nTimeTrace), dtype=np.float32
    )
    detector_positions = np.zeros((len(event_list), nTile, nTile, 3), dtype=np.float32)
    detector_states = np.ones((len(event_list), nTile, nTile), dtype=bool)

    def rufptn_xxyy2sds(rufptn_xxyy_):
        nowIndex = int(np.where(tasd_clf.tasdmc_clf[:, 0] == rufptn_xxyy_)[0])
        return tasd_clf.tasdmc_clf[nowIndex, :]

    for i in range(len(sdmeta_list)):
        # if i != event_index:
        #    continue
        signalMax_xx = 0
        signalMax_yy = 0
        signalMax_size = 0
        firstTime = 10**8
        
        # For filtering in the case
        # when second waveform has sdmeta_list[i][j][1] > 2
        prev_excluded = False
        prev = None
        for j in range(len(sdmeta_list[i])):
            if prev_excluded and (prev == sdmeta_list[i][j][0]):
                prev_excluded = True
                prev = sdmeta_list[i][j][0]
                continue

            if sdmeta_list[i][j][1] <= 2:
                prev_excluded = True
                prev = sdmeta_list[i][j][0]
                continue  ## exclude coincidence signals
            prev_excluded = False
            xx = int(str(int(sdmeta_list[i][j][0])).zfill(4)[:2])
            yy = int(str(int(sdmeta_list[i][j][0])).zfill(4)[2:])
            signal_size = (sdmeta_list[i][j][4] + sdmeta_list[i][j][5]) / 2
            if (sdmeta_list[i][j][2] + sdmeta_list[i][j][3]) / 2 < firstTime:
                firstTime = (sdmeta_list[i][j][2] + sdmeta_list[i][j][3]) / 2
            if signal_size > signalMax_size:
                signalMax_xx = xx
                signalMax_yy = yy
                signalMax_size = signal_size
                # center_j = j
        for k in range(nTile):
            for l in range(nTile):
                xx = signalMax_xx + (k - (nTile - 1) / 2)
                yy = signalMax_yy + (l - (nTile - 1) / 2)
                sdid = round(xx) * 100 + round(yy)
                if sdid not in tasd_clf.tasd_isd_set:  # No corresponding SD exists
                    detector_states[i][k][l] = False
                else:
                    # detector_positions[i][xGrid][yGrid][0] = 1.2 * ((sdmeta_list[i][j][6]-12.2435) - (sdmeta_list[i][center_j][6]-12.2435)) * 1000 # meter
                    # detector_positions[i][xGrid][yGrid][1] = 1.2 * ((sdmeta_list[i][j][7]-16.4406) - (sdmeta_list[i][center_j][7]-16.4406)) * 1000 # meter
                    # detector_positions[i][xGrid][yGrid][2] = 1.2 * (sdmeta_list[i][j][8] - sdmeta_list[i][center_j][8]) * 1000 # meter
                    # sd_clf = rufptn_xxyy2sds(int(sdmeta_list[i][j][0])) / 100  # meter
                    sd_clf = rufptn_xxyy2sds(int(sdid)) / 100  # meter
                    detector_positions[i][k][l][0] = sd_clf[1]
                    detector_positions[i][k][l][1] = sd_clf[2]
                    detector_positions[i][k][l][2] = sd_clf[3]
                    if sdid in badsdinfo_set_list[i]:  ## the status is bad
                        detector_states[i][k][l] = False
                    else:
                        if sdid in [int(row[0]) for row in sdmeta_list[i]]:
                            sdmeta_list_index = int(
                                [
                                    index
                                    for index, sublist in enumerate(sdmeta_list[i])
                                    if sublist[0] == sdid
                                ][0]
                            )
                            if sdmeta_list[i][sdmeta_list_index][1] <= 2:
                                continue  ## exclude coincidence signals

                            arrival_times[i][k][l] = (
                                (
                                    (
                                        sdmeta_list[i][sdmeta_list_index][2]
                                        + sdmeta_list[i][sdmeta_list_index][3]
                                    )
                                    / 2
                                    - firstTime
                                )
                                * 4
                                * 1000
                            )  # nsec

                            fadc_low = np.array(
                                next(
                                    item[3 : 3 + 128]
                                    for item in sdwaveform_list[i]
                                    if item[0] == sdmeta_list[i][sdmeta_list_index][0]
                                )
                            )
                            fadc_up = np.array(
                                next(
                                    item[3 + 128 :]
                                    for item in sdwaveform_list[i]
                                    if item[0] == sdmeta_list[i][sdmeta_list_index][0]
                                )
                            )
                            time_traces[i][k][l][:] = (
                                fadc_low / sdmeta_list[i][sdmeta_list_index][9]
                                + fadc_up / sdmeta_list[i][sdmeta_list_index][10]
                            ) / 2  ## average of lower & upper FADC signal

    end_time = time.time()

    # print calculation time
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds, total events: %d" % (len(event_list)))

    data = dict()
    data["mass_number"] = mass_number
    data["energy"] = energy
    data["xmax"] = xmax
    data["shower_axis"] = shower_axis
    data["shower_core"] = shower_core

    data["arrival_times"] = arrival_times
    data["time_traces"] = time_traces
    data["detector_positions"] = detector_positions
    data["detector_states"] = detector_states

    return data

    ## plot
    if ifPlot:
        import matplotlib.pyplot as plt
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
                next(item[1] for item in sdmeta_list[event_index] if item[0] == wf[0])
                <= 2
            ):  ## plot only space-time cluster SDs
                # alpha = 0.1
                continue
            if wf[0] != prevSD:
                offset += offsetValue
            timeStart = wf[1] / wf[2] * 1.0 * (10**9) / 20  # 1 bin = 20 ns
            ax.plot(
                timeStart + time128bins,
                wf[3 : 3 + 128] + offset,
                color="red",
                alpha=alpha,
            )
            ax.plot(
                timeStart + time128bins,
                wf[3 + 128 :] + offset,
                color="blue",
                alpha=alpha,
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
        if ifSavePlot:
            fig.savefig("Waveform_%s_event_index_%d.png" % (add_label, event_index))

        ## 2) Foot Print
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        sd_show = []
        for j in range(len(sdmeta_list[event_index])):
            if sdmeta_list[event_index][j][1] >= 3:
                sd_show.append(
                    [
                        str(int(sdmeta_list[event_index][j][0])).zfill(4),
                        rufptn_xxyy2sds(int(sdmeta_list[event_index][j][0]))[1] / 100,
                        rufptn_xxyy2sds(int(sdmeta_list[event_index][j][0]))[2] / 100,
                        (
                            sdmeta_list[event_index][j][4]
                            + sdmeta_list[event_index][j][5]
                        )
                        / 2,
                        (
                            sdmeta_list[event_index][j][2]
                            + sdmeta_list[event_index][j][3]
                        )
                        / 2,
                    ]
                )
        sd_show = np.array(sd_show, dtype=float)
        scat = ax.scatter(
            sd_show[:, 1],
            sd_show[:, 2],
            s=np.log10(sd_show[:, 3]) * 100 + 1,
            c=sd_show[:, 4] * 4 * 1000 - min(sd_show[:, 4] * 4 * 1000),
            vmin=0,
            vmax=1.5 * 10**4,
            cmap="rainbow",
        )
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
        if ifSavePlot:
            fig.savefig("FootPrint_%s_event_index_%d.png" % (add_label, event_index))

        ## 3) nTile x nTile SD states
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(nTile):
            for j in range(nTile):
                ax.scatter(
                    i,
                    j,
                    edgecolors=(
                        "black"
                        if np.any(time_traces[event_index][i][j] != 0)
                        else "white"
                    ),
                    c=(
                        arrival_times[event_index][i][j]
                        if np.any(time_traces[event_index][i][j] != 0)
                        else "black"
                    ),
                    cmap="rainbow",
                    vmin=0,
                    vmax=1.5 * 10**4,
                    marker="s" if detector_states[event_index][i][j] else "x",
                    s=100 if np.any(time_traces[event_index][i][j] != 0) else 50,
                )
        plt.gca().set_aspect("equal", adjustable="box")
        cbar = plt.colorbar(scat, ax=ax)
        cbar.set_label(r"relative time [ns]", labelpad=20, rotation=270, fontsize=12)
        ax.set_title(r"%d $\times$ %d SD grids for DNN" % (nTile, nTile))
        ax.set_xlabel("tile index X")
        ax.set_ylabel("tile index Y")
        if ifSavePlot:
            fig.savefig("Tilesfor_DNN_%s_event_index_%d.png" % (add_label, event_index))

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
                    color=(
                        colormap(
                            (arrival_times[event_index][i][j] - 0) / (1.5 * 10**4 - 0)
                        )
                        if np.any(time_traces[event_index][i][j] != 0)
                        else "black"
                    ),
                )
        ax.set_title(r"%d $\times$ %d time traces" % (nTile, nTile))
        ax.set_xlabel("time trace [time bin (20 ns/bin)]")
        ax.set_ylabel("VEM / time bin (float32)")
        if ifSavePlot:
            fig.savefig(
                "TimeTrace_forDNN_%s_event_index_%d.png" % (add_label, event_index)
            )
        ##
        plt.show()
