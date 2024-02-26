from tqdm import tqdm
from dst_content import dst_content
from dst_parsers import dst_sections, shower_params
from dst_parsers import parse_sdmeta, parse_sdwaveform, parse_badsdinfo
from dst_parsers import init_detector_tile, detector_readings

dst_file = "/ceph/work/SATORI/projects/TA-ASIoP/sdanalysis_2018_TALE_TAx4SingleCT_DM/DAT000015_gea.dat.hrspctr.1850.specCuts.dst.gz"


meta_data = dict()
meta_data["interaction_model"]  = "QGSJET-II-03"
meta_data["atmosphere_model"]  = ""
meta_data["emin"]  = ""
meta_data["emax"]  = ""
meta_data["espectrum"]  = "HiRes"
meta_data["DST_file_name"] = dst_file


def parse_file(dst_file):
    dst_string = dst_content(dst_file)

    event_list_str, sdmeta_list_str, sdwaveform_list_str, badsdinfo_list_str = dst_sections(
        dst_string
    )

    mass_number, energy, xmax, shower_axis, shower_core = shower_params(event_list_str)


    sdmeta_list = parse_sdmeta(sdmeta_list_str)
    sdwaveform_list = parse_sdwaveform(sdwaveform_list_str)
    badsdinfo_list = parse_badsdinfo(badsdinfo_list_str)

    nTile = 7  # number of SD per one side
    nTimeTrace = 128  # number of time trace of waveform
    num_events = len(mass_number)
    detector_tile = init_detector_tile(num_events, nTile, nTimeTrace)
    detector_tile = detector_readings(sdmeta_list, sdwaveform_list, detector_tile)


for i in tqdm(range(10),total=10):
    parse_file(dst_file)