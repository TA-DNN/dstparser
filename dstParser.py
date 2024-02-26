import numpy as np
import time
from dst_content import dst_content
from dst_parsers import dst_sections, shower_params
from dst_parsers import parse_sdmeta, parse_sdwaveform, parse_badsdinfo
from dst_parsers import init_detector_tile

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
    print(wform[:, 0])
    wform_xy = wform[:, 0].astype(np.int32)

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
    wform = wform[wform_idx, 3:]

    # print(wform.shape)
    # print(event.shape)
    # input()

    # ix and iy as one array [ix, iy]
    ixy = np.array([event[0] // 100, event[0] % 100])

    # center around detector with max signal
    max_signal_idx = np.argmax((event[4] + event[5]) / 2)
    ixy -= ixy[:, max_signal_idx][:, np.newaxis]

    # cut array size to fit the tile size
    inside_tile = (abs(ixy[0]) < tile_size) & (abs(ixy[0]) < tile_size)
    ixy = ixy[:, inside_tile].astype(np.int32)

    # averaged arrival times
    atimes = (event[2] + event[3]) / 2
    # relative time of first arrived particle
    atimes -= np.min(atimes)

    print(atimes.shape)
    print(arrival_times.shape)

    arrival_times[ievt, ixy[0], ixy[1]] = atimes[inside_tile] * to_nsec

    # wtr = wform.transpose()
    test = wform[:, :128] / event[9] + wform[:, 128:] / event[10]

    print(test.shape)
    input()

    # time_traces[ievt, ixy[0], ixy[1], :] = wforrm[:128]

    # time_traces[i][xGrid][yGrid][:] = (
    #             fadc_low / sdmeta_list[i][j][9] + fadc_up / sdmeta_list[i][j][10]
    #         ) / 2

    input()
    # xx = int(str(int(sdmeta_list[i][j][0])).zfill(4)[:2])
    #     yy = int(str(int(sdmeta_list[i][j][0])).zfill(4)[2:])
    #     xGrid = int(-ix[max_signal_idx] + ix + tile_shift)
    #     yGrid = int(-iy[max_signal_idx] + iy + tile_shift)



elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds, total events: %d" % (len(event_list)))
