from importlib.metadata import version
from dstparser.dst_adapter import parse_dst_file as parse_dst_file_ta
from dstparser.dst_adapter_tax4 import parse_dst_file_tax4


def parse_dst_file(
    dst_file,
    ntile=7,
    xmax_reader=None,
    avg_traces=True,
    add_shower_params=True,
    add_standard_recon=True,
    config=None,
    use_grid_model=True,
    data_type="TA",
):
    if data_type == "TA":
        return parse_dst_file_ta(
            dst_file,
            ntile=ntile,
            xmax_reader=xmax_reader,
            avg_traces=avg_traces,
            add_shower_params=add_shower_params,
            add_standard_recon=add_standard_recon,
            config=config,
            use_grid_model=use_grid_model,
        )
    elif data_type == "TAX4":
        return parse_dst_file_tax4(
            dst_file,
            ntile=ntile,
            xmax_reader=xmax_reader,
            avg_traces=avg_traces,
            add_shower_params=add_shower_params,
            add_standard_recon=add_standard_recon,
            config=config,
            use_grid_model=use_grid_model,
        )
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


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
