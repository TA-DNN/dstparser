## Test init_development 240226

import os
import matplotlib.pyplot as plt
import numpy as np
import sys
os.environ["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"] + ":" + "/dicos_ui_home/anatoli/groupspace/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM/lib"
os.environ["PATH"] = os.environ["PATH"] + ":" + "/dicos_ui_home/anatoli/groupspace/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM/bin"

import subprocess

##
import tasd_clf ## TASD position (CLF)

###
event_index = 1
ifPlot = False


### 
import time
start_time = time.time()

def capture_output(cmd):
    try:
        #output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        output, _ = process.communicate()
        return output
    except subprocess.CalledProcessError as e:
        return e.output


## read DST file
#output = capture_output('sditerator.run ../talesdcalibev_pass2_190502.rufldf.dst.gz').strip().split('\n')[:-1]
output = capture_output('sditerator.run %s'%(sys.argv[1])).strip().split('\n')
#print(output[:10])

## make lists
event_readout = False
sdmeta_readout = False
sdwaveform_readout = False
event_list_str = []
sdmeta_list_str = []
sdwaveform_list_str = []
event = ""
for il, line in enumerate(output):
    if "EVENT DATA" in line:
        event_readout = True
        sdmeta_readout = False
        sdwaveform_readout = False
        continue
    elif "SD meta DATA" in line:
        event_readout = False
        sdmeta_readout = True
        sdwaveform_readout = False
        continue
    elif "SD waveform DATA" in line:
        event_readout = False
        sdmeta_readout = False
        sdwaveform_readout = True
        continue
    if event_readout:
        event_list_str.append(line)
    elif sdmeta_readout:
        sdmeta_list_str.append(line)
    elif sdwaveform_readout:
        sdwaveform_list_str.append(line)


##
## Make Data set
##

# Meta data
Interaction_model = "QGSJET-II-03"
Atmosphere_model = ""
Emin = ""
Emax = ""
Espectrum = "HiRes"
DST_file_name = sys.argv[1]

def CORSIKAparticleID2mass(corsikaPID):
    if corsikaPID == 14:
        return 1
    else:
        return corsikaPID//100

def convert_to_specific_type(columnName, value):
    if columnName == "rusdmc_.parttype":
        return CORSIKAparticleID2mass(int(value))
    elif columnName in ["rusdraw_.yymmdd", "rusdraw_.hhmmss",
                      "rufptn_.nstclust","rusdraw_.nofwf",
                      "rufptn_.xxyy","rufptn_.isgood"]:
        return int(value)
    else:
        return float(value)


# Shower related
event_list = [[float(c) for c in l.split(" ") if c != ""] for l in event_list_str]
"""
event_format = ["mass_number",
                "rusdmc_energy",
                "rusdmc_theta",
                "rusdmc_phi",
                "rusdmc_corexyz[0]",
                "rusdmc_corexyz[1]",
                "rusdmc_corexyz[2]",
                "rusdraw_yymmdd",
                "rusdraw_hhmmss",
                "rufptn_nstclust",
                "rusdraw_nofwf"]
"""
mass_number = np.array([item[0] for item in event_list],
                       dtype = np.int32)
energy = np.array([item[1] for item in event_list],
                         dtype = np.float32)
xmax = np.array([0 for item in event_list])
shower_axis = np.array([[np.sin(item[2]) * np.cos(item[3]),
                         np.sin(item[2]) * np.sin(item[3]),
                         np.cos(item[2])] for item in event_list],
                       dtype = np.float32)
shower_core = np.array([[item[4],
                         item[5],
                         item[6]] for item in event_list],
                         dtype = np.int32)

## Detection related 
sdmeta_list = [[float(c) for c in l.split(" ") if c != ""] for l in sdmeta_list_str]
sdmeta_list = [[[sdmeta_list[i][j+k*11] for j in range(11)] for k in range(len(sdmeta_list[i])//11)] for i in range(len(sdmeta_list))]
sdwaveform_list = [[int(c) for c in l.split(" ") if c != ""] for l in sdwaveform_list_str]
sdwaveform_list = [[[sdwaveform_list[i][j+k*(3+128*2)] for j in range(3+128*2)] for k in range(len(sdwaveform_list[i])//(3+128*2))] for i in range(len(sdwaveform_list))]

# Put largest-signal SD at the center of nTile x nTile grids
nTile = 7 # number of SD per one side
nTimeTrace = 128 # number of time trace of waveform
#
arrival_times = np.zeros((len(event_list),
                          nTile,
                          nTile),
                        dtype = np.float32)
time_traces = np.zeros((len(event_list),
                        nTile,
                        nTile,
                        nTimeTrace),
                       dtype = np.float32)
detector_positions = np.zeros((len(event_list),
                               nTile,
                               nTile,
                               3),
                              dtype = np.float32)
detector_states = np.ones((len(event_list),
                            nTile,
                            nTile),
                           dtype = bool)

def rufptn_xxyy2sds(rufptn_xxyy_):
    nowIndex = int(np.where(tasd_clf.tasdmc_clf[:,0]==rufptn_xxyy_)[0])
    return tasd_clf.tasdmc_clf[nowIndex,:]

for i in range(len(sdmeta_list)):
    #if i>0:
    #    continue
    signalMax_xx = 0
    signalMax_yy = 0
    signalMax_size = 0
    firstTime = 10**8
    for j in range(len(sdmeta_list[i])):
        if sdmeta_list[i][j][1] <= 2:
            continue ## exclude coincidence signals
        xx = int(str(int(sdmeta_list[i][j][0])).zfill(4)[:2])
        yy = int(str(int(sdmeta_list[i][j][0])).zfill(4)[2:])
        signal_size = (sdmeta_list[i][j][4]+sdmeta_list[i][j][5])/2
        if (sdmeta_list[i][j][2]+sdmeta_list[i][j][3])/2 < firstTime:
            firstTime = (sdmeta_list[i][j][2]+sdmeta_list[i][j][3])/2
        if signal_size > signalMax_size:
            signalMax_xx = xx
            signalMax_yy = yy
            signalMax_size = signal_size
            center_j = j
        #print("##",xx,yy,signal_size,signalMax_xx,signalMax_yy,signalMax_size)
    for j in range(len(sdmeta_list[i])):
        if sdmeta_list[i][j][1] <= 2:
            continue ## exclude coincidence signals
        xx = int(str(int(sdmeta_list[i][j][0])).zfill(4)[:2])
        yy = int(str(int(sdmeta_list[i][j][0])).zfill(4)[2:])
        xGrid = int(- signalMax_xx + xx + (nTile-1)/2)
        yGrid = int(- signalMax_yy + yy + (nTile-1)/2)
        #print(xx,yy,xGrid,yGrid)
        if xGrid >= 0 and xGrid < nTile and\
           yGrid >= 0 and yGrid < nTile:
            arrival_times[i][xGrid][yGrid] = ((sdmeta_list[i][j][2] + sdmeta_list[i][j][3])/2 - firstTime) * 4 * 1000 # nsec
            fadc_low = np.array(next(item[3:3+128] for item in sdwaveform_list[i] if item[0] == sdmeta_list[i][j][0]))
            fadc_up  = np.array(next(item[3+128:]  for item in sdwaveform_list[i] if item[0] == sdmeta_list[i][j][0]))
            time_traces[i][xGrid][yGrid][:] = (fadc_low/sdmeta_list[i][j][9]+ fadc_up/sdmeta_list[i][j][10]) / 2 ## average of lower & upper FADC signal
            #detector_positions[i][xGrid][yGrid][0] = 1.2 * ((sdmeta_list[i][j][6]-12.2435) - (sdmeta_list[i][center_j][6]-12.2435)) * 1000 # meter
            #detector_positions[i][xGrid][yGrid][1] = 1.2 * ((sdmeta_list[i][j][7]-16.4406) - (sdmeta_list[i][center_j][7]-16.4406)) * 1000 # meter
            #detector_positions[i][xGrid][yGrid][2] = 1.2 * (sdmeta_list[i][j][8] - sdmeta_list[i][center_j][8]) * 1000 # meter
            sd_clf = rufptn_xxyy2sds(int(sdmeta_list[i][j][0]))/100 # meter
            detector_positions[i][xGrid][yGrid][0] = sd_clf[1]
            detector_positions[i][xGrid][yGrid][1] = sd_clf[2]
            detector_positions[i][xGrid][yGrid][2] = sd_clf[3]
            detector_states[i][xGrid][yGrid] = True

end_time = time.time()

# print calculation time
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds, total events: %d"%(len(event_list)))



## plot
if ifPlot:
    from matplotlib.colors import LogNorm
    
    ## 1) Waveform
    fig = plt.figure()
    ax=fig.add_subplot(1,1,1)
    time128bins=np.arange(0,128,1)
    offsetValue = 300
    offset = 0
    prevSD = -1 * offsetValue
    for wf in np.array(sdwaveform_list[event_index]):
        alpha = 1
        if next(item[1] for item in sdmeta_list[event_index] if item[0] == wf[0]) <= 2: ## plot only space-time cluster SDs
            #alpha = 0.1
            continue
        if wf[0] != prevSD:
            offset += offsetValue
        timeStart = wf[1]/wf[2] * 1. * (10**9) / 20 # 1 bin = 20 ns
        ax.plot(timeStart + time128bins,
                wf[3:3+128]+offset,color="red", alpha = alpha)
        ax.plot(timeStart + time128bins,
                wf[3+128:]+offset,color="blue", alpha = alpha)
        ax.text(timeStart, offset, wf[0])
        prevSD = wf[0]
    ax.set_xlabel("time bin [20 ns]")
    ax.set_ylabel("FADC count")
    ax.set_title(r"$E_{gen}$ = %.2f EeV, $\theta_{gen}$ = %.2f deg, $\phi_{gen}$ = %.2f deg"%(event_list[event_index][1], event_list[event_index][2]*180/np.pi, event_list[event_index][3]*180/np.pi))
    fig.savefig("Waveform_event_index_%d.png"%(event_index))
    
    ## 2) Foot Print
    fig = plt.figure()
    ax=fig.add_subplot(1,1,1)
    sd_show = []
    for j in range(len(sdmeta_list[event_index])):
        if sdmeta_list[event_index][j][1] > 2:
            sd_show.append([str(int(sdmeta_list[event_index][j][0])).zfill(4),
                            rufptn_xxyy2sds(int(sdmeta_list[event_index][j][0]))[1]/100,
                            rufptn_xxyy2sds(int(sdmeta_list[event_index][j][0]))[2]/100,
                            (sdmeta_list[event_index][j][4] + sdmeta_list[event_index][j][5])/2,
                            (sdmeta_list[event_index][j][2] + sdmeta_list[event_index][j][3])/2])
    sd_show = np.array(sd_show,dtype=float)
    print(sd_show)
    scat = ax.scatter(sd_show[:,1],
                      sd_show[:,2],
                      s = np.log10(sd_show[:,3]) * 100 + 1,
                      c = sd_show[:,4] * 4 * 1000 - min(sd_show[:,4] * 4 * 1000),
                      vmin = 0,
                      vmax = 10**4,
                      cmap = "rainbow")
    #vmin = min(sd_show[:,4]),
    #vmax = max(sd_show[:,4]))
    cbar = plt.colorbar(scat, ax = ax)
    cbar.set_label(r'relative time [ns]',
                   labelpad=20, rotation=270, fontsize=12)
    for i in range(tasd_clf.tasdmc_clf.shape[0]):
        ax.scatter(tasd_clf.tasdmc_clf[i,1]/100,
                   tasd_clf.tasdmc_clf[i,2]/100,
                   marker = "s",
                   c = "white",
                   s = 3,
                   edgecolors = "black")
    plt.gca().set_aspect('equal', adjustable='box')
    ax.scatter(event_list[event_index][4]/100,
               event_list[event_index][5]/100,
               marker = "*",
               color = "white",
               edgecolors = "black")
    ax.set_xlabel("CLF-x [m]")
    ax.set_ylabel("CLF-y [m]")
    fig.savefig("FootPrint_event_index_%d.png"%(event_index))
    
    ## 3) nTile x nTile SD states
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(nTile):
        for j in range(nTile):
            ax.scatter(i,
                       j,
                       edgecolors = "black" if np.any(time_traces[event_index][i][j] != 0) else "white",
                       c = arrival_times[event_index][i][j] if np.any(time_traces[event_index][i][j] != 0) else "black",
                       cmap = "rainbow",
                       vmin = 0,
                       vmax = 10**4,
                       marker = "s" if detector_states[event_index][i][j] else "x",
                       s = 100 if np.any(time_traces[event_index][i][j] != 0) else 50)
    plt.gca().set_aspect('equal', adjustable='box')
    #scat = ax.scatter([],[],
    #           c = [],
    #           cmap = "rainbow",
    #          vmin = 0,
    #          vmax = 10**4)
    cbar = plt.colorbar(scat, ax = ax)
    cbar.set_label(r'relative time [ns]',
                   labelpad=20, rotation=270, fontsize=12)
    ax.set_title(r"%d $\times$ %d SD grids for DNN"%(nTile,nTile))
    ax.set_xlabel("tile index X")
    ax.set_ylabel("tile index Y")
    fig.savefig("Tilesfor_DNN_event_index_%d.png"%(event_index))

    ## 4) nTile x nTile time trace
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    colormap = plt.get_cmap("rainbow")
    step = 10
    for i in range(nTile):
        for j in range(nTile):
            ax.step(np.arange(nTimeTrace),
                    time_traces[event_index][i][j] + step * (i * nTile + j),
                    where = "mid",
                    #c = arrival_times[event_index][i][j],
                    color = colormap((arrival_times[event_index][i][j] - 0) / (10**4 - 0)) if np.any(time_traces[event_index][i][j] != 0) else "black")
    ax.set_title(r"%d $\times$ %d time traces"%(nTile,nTile))
    ax.set_xlabel("time trace [time bin (20 ns/bin)]")
    ax.set_ylabel("VEM / time bin (float32)")
    fig.savefig("TimeTrace_forDNN_event_index_%d.png"%(event_index))
    ##
    plt.show()



            

"""
sdmeta_format = ["rufptn_.xxyy",
                 "rufptn_.isgood",
                 "rufptn_.reltime[0]",
                 "rufptn_.reltime[1]",
                 "rufptn_.pulsa[0]",
                 "rufptn_.pulsa[1]",
                 "rufptn_.xyzclf[0]",
                 "rufptn_.xyzclf[1]",
"rufptn_.xyzclf[2]",
"rufptn_.vem[0]",
"rufptn_.vem[1]"]
sdmeta_dict = [
    [
        {
            sdmeta_format[k]: int(sdmeta_list[i][j][k]) if sdmeta_format[k] in ["rufptn_.xxyy", "rufptn_.isgood"] else float(sdmeta_list[i][j][k]) 
            for k in range(len(sdmeta_format))
        } 
        for j in range(len(sdmeta_list[i]))
    ] 
    for i in range(len(sdmeta_list))
]
"""


## sd waveform data
"""
sdwaveform_format = ["rusdraw_.xxyy",
                     "rusdraw_.clkcnt",
                     "rusdraw_.mclkcnt",
                     "rusdraw_.fadc[0]",
                     "rusdraw_.fadc[1]"]
sdwaveform_dict = [
    [
        {
            sdwaveform_format[0]: int(sdwaveform_list[i][j][0]),
            sdwaveform_format[1]: int(sdwaveform_list[i][j][1]),
            sdwaveform_format[2]: int(sdwaveform_list[i][j][2]),
            #f"{sdwaveform_format[3]}[{k}]": int(sdwaveform_list[i][j][3 + k]),
            #f"{sdwaveform_format[4]}[{k}]": int(sdwaveform_list[i][j][3 + 128 + k])
            sdwaveform_format[3]: [int(sdwaveform_list[i][j][3 + k]) for k in range(128)],
            sdwaveform_format[4]: [int(sdwaveform_list[i][j][3 + 128 + k]) for k in range(128)]
        }
        for j in range(len(sdwaveform_list[i]))
        #for k in range(128)
    ]
    for i in range(len(sdwaveform_list))
]
"""

