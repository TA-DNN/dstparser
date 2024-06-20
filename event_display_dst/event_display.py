import numpy as np
import matplotlib.pyplot as plt
from dstparser.tasd_clf import tasdmc_clf
from dstparser.dst_adapter import get_sd_position

def exclude_signal(isgood,
                   isgood_option):
    ## excluding not-shower-related signals
    if isgood_option == "nstclust" and isgood <= 3:
        return True
    ## excluding not-shower-related signals & saturated signals
    elif isgood_option == "nstclust_no_saturate" and (isgood <= 3 or isgood == 5):
        return True
    ## excluding signals recorded by bad SDs (bad pedstal fluctuaation, bad GPS status, ...)
    elif isgood_option == "not_bad" and isgood == 0:
        return True
    ## excluding not-shower-related signals, but not excluding signals recorded by bad SDs
    elif isgood_option == "nstclust_with_bad" and (1 <= isgood <= 3):
        return True
    else:
        return False

def scatter_footprint(ax,
                       signal_position_x,
                       signal_position_y,
                       signal_size,
                       signal_time,
                       ):
    signal_time_to_musec = 1200 / (3 * 10**8) *(10**6)
    scat = ax.scatter(signal_position_x,
                      signal_position_y,
                      s = np.log10(signal_size)*200,
                      c = signal_time * signal_time_to_musec,
                     linestyles = "-",
                     linewidth  = 1.5,
                     edgecolors="black",
                     cmap="rainbow",
                     alpha=0.8)
    colors_now = np.array([scat.to_rgba(ii*4) for ii in signal_time])
    cbar = plt.colorbar(scat, ax = ax)
    cbar.set_label(r'relative time [$\mu$s]',
                    labelpad=20,rotation=270,fontsize=16)
    ax.set_xlabel('X [km]',fontsize=16)
    ax.set_ylabel('Y [km]',fontsize=16)
    return colors_now

def footprint(data,
              index, 
              isgood_option = False):
    cm_to_km = 1 / (10**5)
    m_to_km = 1 / (10**3)
    signals = []
    for i, sdid in enumerate(data["signal_ixy"][index]):
        if isgood_option:
            isgood = data["signal_isgood"][index][i]
            if exclude_signal(isgood,
                              isgood_option):
                continue
        signal_positions = get_sd_position(sdid)
        signals.append([signal_positions[0][0] * m_to_km,
                        signal_positions[0][1] * m_to_km,
                        (data["signal_total_low"][index][i] + data["signal_total_up"][index][i]) / 2,
                        (data["arrival_times_low"][index][i] + data["arrival_times_up"][index][i]) / 2,
                        sdid])
    signals = np.array(signals)
    fig=plt.figure(figsize=(6,6))
    fig.canvas.draw()
    ax = fig.add_subplot(1,1,1)
    colors = scatter_footprint(ax,
                                signals[:,0],
                                signals[:,1],
                                signals[:,2],
                                signals[:,3]
                                )
    for i, sd in enumerate(tasdmc_clf[:,3]):
        ax.scatter(tasdmc_clf[i,1] * cm_to_km,
                   tasdmc_clf[i,2] * cm_to_km,
                   edgecolor="black",
                   color="black",
                   marker="s",
                   s=10)
        # plt.annotate(int(sd),
        #              xy=(tasdmc_clf[i,1] * cm_to_km,
        #                  tasdmc_clf[i,2] * cm_to_km),
        #              size=7)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.tick_params(labelsize=16)
    return fig, ax, colors, signals[:,4]

def waveforms(data,
              index, 
              colors = False,
              sdid_colors = False,
              with_start_time=True,
              isgood_option = False):
    ns_to_time_slice = 1. * 10**9 / 20
    n_of_timebins = 128
    timeslice_to_musec = 1 / 50
    t = np.arange(0, n_of_timebins, 1)
    wf_with_time = []

    count_wf_dict = {val: 0 for val in data["signal_ixy"][index]}
    indices_wf = []
    for i, sdid in enumerate(data["wf_ixy"][index]):
        if isgood_option:
            tmp_ind = data["signal_ixy"][index].index(sdid, count_wf_dict[sdid])
            indices_wf.append(tmp_ind)
            count_wf_dict[sdid] += 1
            isgood = data["signal_isgood"][index][tmp_ind]
            if exclude_signal(isgood,
                              isgood_option):
                continue
        wf_with_time.append([data["wf_clock"][index][i] / data["wf_max_clock"][index][i] * ns_to_time_slice,
                             data["fadc_low"][index][i],
                             data["fadc_up"][index][i],
                             sdid])
        

    wf_with_time = sorted(wf_with_time)

    already_written_sd = []
    already_written_sd_accum = []
    unique_count = len(set(wf_with_time[k][3] for k in range(len(wf_with_time))))
    start_time = min(wf_with_time[k][0] for k in range(len(wf_with_time)))
    last_time = max(wf_with_time[k][0] + n_of_timebins for k in range(len(wf_with_time)))

    fig, ax = plt.subplots(unique_count, 1, figsize=(6, 0.8 * unique_count))
    if unique_count == 1:
        ax = [ax]
    cm = plt.get_cmap("jet")

    kk = 0
    for k in range(len(wf_with_time)):
        linestyle_tmp = "-"
        alpha_tmp = 1
        wf_start = wf_with_time[k][0] if with_start_time else min(wf_with_time[i][0] for i in range(len(wf_with_time)))

        if wf_with_time[k][3] in already_written_sd:
            appearance_time = already_written_sd_accum.count(wf_with_time[k][3])
            if not with_start_time:
                wf_start += n_of_timebins * appearance_time
            color_tmp = colors[sdid_colors == wf_with_time[k][3]] if (colors.any() and sdid_colors.any()) else "black"
            tmp_ind = already_written_sd.index(wf_with_time[k][3])
        else:
            tmp_ind = kk
            color_tmp = colors[sdid_colors == wf_with_time[k][3]] if (colors.any() and sdid_colors.any()) else "black"
            already_written_sd.append(wf_with_time[k][3])
            kk += 1

        ax[tmp_ind].step((t + wf_start - start_time) * timeslice_to_musec,
                          np.array(wf_with_time[k][1]),
                          color=color_tmp,
                          where="mid", 
                          linestyle=linestyle_tmp,
                          lw=0.8, 
                          alpha=alpha_tmp - 0.4)
        ax[tmp_ind].step((t + wf_start - start_time) * timeslice_to_musec, 
                         np.array(wf_with_time[k][2]), 
                         color=color_tmp,
                         where="mid", 
                         linestyle=linestyle_tmp, alpha=alpha_tmp)
        ax[tmp_ind].text(0.95,
                         0.95,
                         "SD%04d" % wf_with_time[k][3],
                         transform=ax[tmp_ind].transAxes,
                         verticalalignment='top', 
                         horizontalalignment='right', 
                         fontsize=10)
        already_written_sd_accum.append(wf_with_time[k][3])

        if tmp_ind < unique_count - 1:
            ax[tmp_ind].tick_params(labelbottom=False)

    xlabel = "Relative time from the first waveform [$\mu$s]" if with_start_time else "time [$\mu$s]"
    ax[-1].set_xlabel(xlabel, fontsize=12)
    x_min = -1 * timeslice_to_musec if not with_start_time else -1
    x_max = (last_time - start_time) * timeslice_to_musec + 0.5 if with_start_time else n_of_timebins * max(already_written_sd_accum.count(wf) for wf in set(wf_with_time[i][3] for i in range(len(wf_with_time)))) * timeslice_to_musec

    for ax_ in ax:
        ax_.set_xlim(x_min, x_max)

    fig.text(0.02, 0.5, 'FADC count', va='center', rotation='vertical', fontsize=12)
    fig.tight_layout(rect=[0.05, 0.05, 1.1, 1])

    for ax_ in ax:
        ax_.tick_params(labelsize=10)

    return fig, ax
