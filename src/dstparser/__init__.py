from importlib.metadata import version
from dstparser.dst_adapter import parse_dst_file

__version__ = version("dstparser")


__all__ = [
    "parse_dst_file",
    "dst_adapter",
    "dst_parsers",
    "dst_reader",
    "env_vars",
    "plots",
    "paths",
    "__version__",
]
