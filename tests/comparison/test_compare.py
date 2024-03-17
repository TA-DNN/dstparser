from tqdm import tqdm
import numpy as np
from dstparser import parse_dst_file
from dstParser import parse_script

dst_file = "/ceph/work/SATORI/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM/DAT000015_gea.dat.hrspctr.1850.specCuts.dst.gz"

data0 = parse_dst_file(dst_file)
data1 = parse_script(dst_file)

for key in data1:
    print(key, data0[key].shape, data1[key].shape, np.allclose(data0[key], data1[key], atol=1e-5))

# import plots
# import matplotlib.pyplot as plt
# fig_wf, ax_wf = plots.waveform_plot(data0, 0)
# fig_tile, ax_tile = plots.tile_plot(data0, 0)
# plt.show()

# for key, val in data.items():
#     if isinstance(val, np.ndarray):
#         print(key, val.shape)
#     else:
#         print(key, len(val), val)

# for ii in data["detector_states"]:
#     print(ii)
# for i in tqdm(range(10), total=10):
#     parse_dst_file(dst_file)
