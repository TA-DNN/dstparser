from importlib.metadata import version
from dstparser.dst_adapter import parse_dst_file
from dstparser.dst_adapter_tax4 import parse_dst_file_tax4

__version__ = version("dstparser")


__all__ = [
    "parse_dst_file",
    "parse_dst_file_tax4",
    "dst_adapter",
    "dst_parsers",
    "dst_reader",
    "env_vars",
    "plots",
    "paths",
    "__version__",
]
