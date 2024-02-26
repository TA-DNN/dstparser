import numpy as np
import tasd_clf


def CORSIKAparticleID2mass(corsikaPID):
    if corsikaPID == 14:
        return 1
    else:
        return corsikaPID // 100
    
    
def convert_to_specific_type(columnName, value):
    if columnName == "rusdmc_.parttype":
        return CORSIKAparticleID2mass(int(value))
    elif columnName in [
        "rusdraw_.yymmdd",
        "rusdraw_.hhmmss",
        "rufptn_.nstclust",
        "rusdraw_.nofwf",
        "rufptn_.xxyy",
        "rufptn_.isgood",
    ]:
        return int(value)
    else:
        return float(value)
    
    
def rufptn_xxyy2sds(rufptn_xxyy_):
    nowIndex = int(np.where(tasd_clf.tasdmc_clf[:, 0] == rufptn_xxyy_)[0])
    return tasd_clf.tasdmc_clf[nowIndex, :]        