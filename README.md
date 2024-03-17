# dstparser

Reads *dst TA files and fills dictionary with data required for DNN training

Installation:
`pip install -e .`

Usage:
```python
from dstparser import parse_dst_file

dst_file = "/path/to/dst/file.dst.gz"
data = parse_dst_file(dst_file, up_low_traces=True)

# Usa data dict to dump to hdf5 file
# Example in dstparser/slurm
```    