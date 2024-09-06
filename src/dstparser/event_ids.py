

data_set_root = "/ceph/work/SATORI/projects/TA-ASIoP/tasdmc_dstbank/"
data_set_base = dict()
data_set_base[data_set_root + "qgsii04helium/080417_160603/Em1_bsdinfo"] = 10040001




def add_event_ids(data, filename):
    """Write additional information retrived from file name"""
    ifname = Path(filename).parts[-1]
    ifname_parts = re.split(r"[_,.\s]", ifname)

    keys = []
    values = []
    # TA DATA
    if ifname_parts[0].startswith("tasdcalibev"):
        key, value = "ta_obs_date", int(ifname_parts[2])
        keys.append(key)
        values.append(value)
    # TA MC
    elif ifname_parts[0].startswith("DAT"):
        key, value = "corsika_shower_id", int(ifname_parts[0][3:7])
        keys.append(key)
        values.append(value)
        print("int(ifname_parts[0][7:]) = ", int(ifname_parts[0][7:]))
        key, value = "energy_bin_id", int(ifname_parts[0][7:9])
        keys.append(key)
        values.append(value)

    else:
        key, value = None, None

    if len(keys) > 0:
        data_len = next(iter(data.values())).shape[0]
        for key, value in zip(keys, values):
            data[key] = np.full((data_len,), value, dtype=np.int64)
    return data
