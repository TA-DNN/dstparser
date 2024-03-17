from tqdm import tqdm
import numpy as np
from dst_parsers import parse_dst_file
from dstParser import parse_script
from xmax_reader import XmaxReader
from read_data import data_files

dir_with_dst = "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/qgsii03proton/080511_230511/noCuts_HiResSpectrum/"

dst_files = data_files(data_dir=dir_with_dst, glob_pattern="**/DAT*gz")

dst_file = str(dst_files[1885])

xmax_reader = XmaxReader()
data0 = parse_dst_file(dst_file, xmax_reader)
# data1 = parse_script(dst_file)

# for key in data1:
#     print(
#         key,
#         data0[key].shape,
#         data1[key].shape,
#         np.allclose(data0[key], data1[key], atol=1e-7),
#     )


for key, val in data0.items():
    if isinstance(val, np.ndarray):
        print(key, val.shape)
    else:
        print(key, len(val), val)

print(data0["energy"])
print(data0["xmax"])
# for ii in data["detector_states"]:
#     print(ii)
# for i in tqdm(range(10), total=10):
#     parse_dst_file(dst_file)
