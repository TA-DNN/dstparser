import numpy as np
import time
from dst_content import dst_content
from dst_parsers import dst_sections, shower_params
from dst_parsers import parse_sdmeta, parse_sdwaveform, parse_badsdinfo
from dst_parsers import init_detector_tile
from utils import  rufptn_xxyy2sds, find_pos, tile_positions

dst_file = "/ceph/work/SATORI/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM/DAT000015_gea.dat.hrspctr.1850.specCuts.dst.gz"


meta_data = dict()
meta_data["interaction_model"]  = "QGSJET-II-03"
meta_data["atmosphere_model"]  = ""
meta_data["emin"]  = ""
meta_data["emax"]  = ""
meta_data["espectrum"]  = "HiRes"
meta_data["DST_file_name"] = dst_file

start_time = time.time()


dst_string = dst_content(dst_file)

event_list_str, sdmeta_list_str, sdwaveform_list_str, badsdinfo_list_str = dst_sections(
    dst_string
)

mass_number, energy, xmax, shower_axis, shower_core = shower_params(event_list_str)


sdmeta_list = parse_sdmeta(sdmeta_list_str)
sdwaveform_list = parse_sdwaveform(sdwaveform_list_str)
badsdinfo_list = parse_badsdinfo(badsdinfo_list_str)



num_events = len(mass_number)
detector_tile = init_detector_tile(num_events)

nTile = detector_tile["arrival_times"].shape[1]
tile_size = (nTile - 1) / 2 + 1
to_nsec = 4 * 1000

for ievt, (event, wform) in enumerate(zip(sdmeta_list, sdwaveform_list)):
    # Events with > 2 detectors

    print(event.shape, wform.shape)
    print(event[0])
    print(wform[0])
    wform_xy = wform[0].astype(np.int32)

    # Number of indecies of waveforms
    # corresponding to specific detector with index idetector
    wform_idx = dict()
    wform_idx = []
    for idetector, xycoord in enumerate(event[0]):
        # wform_idx[idetector] = np.where(wform_xy == xycoord)[0]
        wform_idx.append(np.where(wform_xy == xycoord)[0])

    # wform[wform_idx[][0]]
    # Print the result
    # for key, value in wform_idx.items():
    #     print(key, value)
    # print(sdwaveform_list[ievt].transpose()[0])

    # print(sdwaveform_list[ievt].transpose()[3:,0])
    # input()

    # idetectors = np.where(event[1] > 2)[0]
    event = event[:, event[1] > 2]

    wform_idx = []
    for ievt, xycoord in enumerate(event[0]):
        # Take only the first waveform (second [0])
        wform_idx.append(np.where(wform_xy == xycoord)[0][0])

    # wform = wform[wform_idx[idetectors][0]][3:]
    # print(wform_idx)
    # print(idetectors)
    # print(wform_idx[list(idetectors)][0])
    # input()
    wform = wform[3:,wform_idx]

    # print(wform.shape)
    # print(event.shape)
    # input()

    # ix and iy as one array [ix, iy]
    ixy = np.array([event[0] // 100, event[0] % 100])

    # print(ixy)
    # center around detector with max signal
    max_signal_idx = np.argmax((event[4] + event[5]) / 2)
    
    # Indicies of central detector
    ixy0 = ixy[:, max_signal_idx]
    detector_tile["detector_positions"][ievt] = tile_positions(ixy0, nTile)
    
    # print(detector_tile["detector_positions"][ievt, 0, 0])
    # print(detector_tile["detector_positions"][ievt, 6, 6])
    # input()
    
    ixy -= ixy0[:, np.newaxis]
    
    # print(ixy)
    # input()
    # cut array size to fit the tile size
    inside_tile = (abs(ixy[0]) < tile_size) & (abs(ixy[0]) < tile_size)
    ixy = ixy[:, inside_tile].astype(np.int32)

    # averaged arrival times
    atimes = (event[2] + event[3]) / 2
    # relative time of first arrived particle
    atimes -= np.min(atimes)

    print(atimes.shape)
    print(detector_tile["arrival_times"].shape)
    print(f"ixy.shape = {ixy.shape}")
    print(f"wform.shape = {wform.shape}")

    detector_tile["arrival_times"][ievt, ixy[0], ixy[1]] = atimes[inside_tile] * to_nsec

    ttrace = (wform[:128] / event[9] + wform[128:] / event[10])/2
    detector_tile["time_traces"][ievt, ixy[0], ixy[1], :] = ttrace.transpose()
    
    # print(detector_tile["time_traces"][ievt])

    # print(f"test.shape = {test.shape}")
    # ddd = detector_tile["time_traces"][ievt, ixy[0], ixy[1],:]
    # print(f"test.shape = {ddd.shape}")
    # # print(test.shape)
    # detector_tile["detector_positions"][ievt, ixy[0], ixy[1]] = find_pos(event[0])
    # print(detector_tile["detector_positions"][ievt, 0, :])
    # input()
    # res = rufptn_xxyy2sds(event[0]) / 100
    # print(res)
    # input()

    # time_traces[ievt, ixy[0], ixy[1], :] = wforrm[:128]

    # time_traces[i][xGrid][yGrid][:] = (
    #             fadc_low / sdmeta_list[i][j][9] + fadc_up / sdmeta_list[i][j][10]
    #         ) / 2

    # input()
    # xx = int(str(int(sdmeta_list[i][j][0])).zfill(4)[:2])
    #     yy = int(str(int(sdmeta_list[i][j][0])).zfill(4)[2:])
    #     xGrid = int(-ix[max_signal_idx] + ix + tile_shift)
    #     yGrid = int(-iy[max_signal_idx] + iy + tile_shift)



# elapsed_time = end_time - start_time
# print(f"Elapsed Time: {elapsed_time} seconds, total events: %d" % (len(event_list)))
