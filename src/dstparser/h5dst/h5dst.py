import h5py
import numpy as np
from tqdm.auto import tqdm


class H5DST:
    def __init__(self, filename):
        self.file = h5py.File(filename, "r")
        self._collect_info()

    def show(self):
        def walk(name, obj, indent=0):
            pad = "  " * indent
            if isinstance(obj, h5py.Group):
                print(f"{pad}{name}")
                for key in obj:
                    walk(f"{key}", obj[key], indent + 1)
            elif isinstance(obj, h5py.Dataset):
                print(f"{pad}{name}{obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Datatype):
                print(f"{pad}{name}")

        walk("", self.file["/"], indent=-1)

    def keys(self):
        return self._data.keys()

    def _collect_info(self):
        self._data = dict()

        def walk(name, obj, indent=0):
            if isinstance(obj, h5py.Group):
                for key in obj:
                    walk(f"{name}/{key}", obj[key], indent + 1)
            elif isinstance(obj, h5py.Dataset):
                shape0 = obj.shape[0]

                if shape0 == self.file["arrival_times"].shape[0]:
                    self._data[name[1:]] = HitLevel(
                        self.file["hit_offsets"], self.file[name[1:]]
                    )
                elif shape0 == self.file["time_traces"].shape[0]:
                    self._data[name[1:]] = TTLevel(self.file)
                else:
                    self._data[name[1:]] = self.file[name[1:]]

        walk("", self.file["/"], indent=-1)

    def __getitem__(self, key):
        return self._data[key]

    def __del__(self):
        if hasattr(self, "file") and self.file is not None:
            try:
                self.file.close()
            except:
                pass


class HitLevel:
    def __init__(self, hit_offsets, data):
        self.hit_offsets = hit_offsets
        self.data = data

    def __getitem__(self, event_idx):
        if isinstance(event_idx, slice):
            ev1 = event_idx.start
            ev2 = event_idx.stop
        else:
            ev1 = event_idx
            ev2 = event_idx + 1

        return self.data[self.hit_offsets[ev1] : self.hit_offsets[ev2]]


class ListWLen(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.shape = len(self)


class TTLevel:
    def __init__(self, file):
        self.f = file
        self.hit_offsets = self.f["hit_offsets"]
        self.hit_tt_offsets = self.f["hit_tt_offsets"]
        self.nfolds = self.f["nfold"]
        self.ttraces = self.f["time_traces"]

    def __getitem__(self, idx):

        if isinstance(idx, int):
            event_idx = idx
            hit_idx = None
        elif len(idx) == 2:
            event_idx = idx[0]
            hit_idx = idx[1]

        start_hit = self.hit_offsets[event_idx]
        end_hit = self.hit_offsets[event_idx + 1]

        # Per-event time-trace slice
        start_tt = self.hit_tt_offsets[start_hit]
        end_tt = self.hit_tt_offsets[end_hit]

        counts = self.nfolds[start_hit:end_hit]
        res = ListWLen(np.split(self.ttraces[start_tt:end_tt], np.cumsum(counts)[:-1]))
        if hit_idx is not None:
            res = res[hit_idx]
        return res


class MultiH5DST:
    def __init__(self, filenames):
        if not isinstance(filenames, list):
            filenames = [filenames]

        self.filenames = filenames
        self.files = [h5py.File(fn, "r") for fn in filenames]

        self.event_counts = [f["energy"].shape[0] for f in self.files]
        self.hit_counts = [f["arrival_times"].shape[0] for f in self.files]
        self.tt_counts = [f["time_traces"].shape[0] for f in self.files]

        self.event_offsets = np.cumsum([0] + self.event_counts)
        self.hit_offsets = np.cumsum([0] + self.hit_counts)
        self.tt_offsets = np.cumsum([0] + self.tt_counts)

        self.nevents = self.event_offsets[-1]

        self._collect_info()

    def __len__(self):
        return self.nevents

    def _collect_info(self):
        self._data = dict()

        def walk(name, obj, indent=0):
            if isinstance(obj, h5py.Group):
                for key in obj:
                    walk(f"{name}/{key}", obj[key], indent + 1)
            elif isinstance(obj, h5py.Dataset):
                shape0 = obj.shape[0]

                if shape0 == self.files[0]["energy"].shape[0]:
                    self._data[name[1:]] = EventLevel(self, name[1:])

        walk("", self.files[0]["/"], indent=-1)

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]

    def show(self, file_idx=0):
        def walk(name, obj, indent=0):
            pad = "  " * indent
            if isinstance(obj, h5py.Group):
                print(f"{pad}{name}")
                for key in obj:
                    walk(f"{key}", obj[key], indent + 1)
            elif isinstance(obj, h5py.Dataset):
                print(f"{pad}{name}{obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Datatype):
                print(f"{pad}{name}")

        print(self.filenames[file_idx])
        walk("", self.files[file_idx]["/"], indent=-1)

    class _FilesAccessor:
        def __init__(self, parent):
            self.parent = parent

        def __getitem__(self, key):
            if isinstance(key, int):
                fi = np.searchsorted(self.parent.event_offsets, key + 1) - 1
                local = key - self.parent.event_offsets[fi]
                return [(int(fi), self.parent.files[fi].filename, int(local))]

            elif isinstance(key, slice):
                start, stop, step = key.indices(self.parent.event_offsets[-1])
                if step != 1:
                    raise NotImplementedError("Only step=1 supported")

                f_start = np.searchsorted(self.parent.event_offsets, start + 1) - 1
                f_stop = np.searchsorted(self.parent.event_offsets, stop) - 1

                out = []
                for fi in range(f_start, f_stop + 1):
                    local_start = 0
                    local_stop = (
                        self.parent.event_offsets[fi + 1]
                        - self.parent.event_offsets[fi]
                    )
                    if fi == f_start:
                        local_start = start - self.parent.event_offsets[fi]
                    if fi == f_stop:
                        local_stop = stop - self.parent.event_offsets[fi]
                    out.append(
                        (
                            int(fi),
                            self.parent.files[fi].filename,
                            slice(int(local_start), int(local_stop)),
                        )
                    )
                return out

    @property
    def from_which_file(self):
        return self._FilesAccessor(self)

    def close(self):
        for f in self.files:
            f.close()


class EventLevel:
    def __init__(self, mdata, array):
        self.mdata = mdata
        self.array = array

    @property
    def shape(self):
        return (len(self.mdata), *self.mdata.files[0][self.array].shape[1:])

    def __getitem__(self, event_idx):
        mdata = self.mdata
        array = self.array
        if isinstance(event_idx, int):
            file_idx = np.searchsorted(mdata.event_offsets, event_idx + 1) - 1
            local_event = event_idx - mdata.event_offsets[file_idx]
            f = mdata.files[file_idx]
            return f[array][local_event : local_event + 1]

        if isinstance(event_idx, slice):
            start, stop, step = event_idx.indices(mdata.event_offsets[-1])
            if step != 1:
                raise NotImplementedError("Only step=1 supported")

            f_start = np.searchsorted(mdata.event_offsets, start + 1) - 1
            f_stop = np.searchsorted(mdata.event_offsets, stop) - 1

            if f_start == f_stop:
                f = mdata.files[f_start]
                local_start = start - mdata.event_offsets[f_start]
                local_stop = stop - mdata.event_offsets[f_start]
                return f[array][local_start:local_stop]

            # spanning multiple files
            total_len = stop - start
            sample = mdata.files[f_start][array][0:1]
            out = np.empty((total_len,) + sample.shape[1:], dtype=sample.dtype)

            pos = 0
            for fi in tqdm(
                range(f_start, f_stop + 1), total=len(range(f_start, f_stop + 1))
            ):
                f = mdata.files[fi]
                local_start = 0
                local_stop = mdata.event_offsets[fi + 1] - mdata.event_offsets[fi]
                if fi == f_start:
                    local_start = start - mdata.event_offsets[fi]
                if fi == f_stop:
                    local_stop = stop - mdata.event_offsets[fi]
                if local_start < local_stop:
                    block = f[array][local_start:local_stop]
                    out[pos : pos + len(block)] = block
                    pos += len(block)
            return out
