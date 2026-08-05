import numpy as np
from pathlib import Path


def find_files(dirs, globs):

    def convert_to_list(value):
        if isinstance(value, list):
            return value
        elif isinstance(value, str) and "," in value:
            return [item.strip() for item in value.split(",")]
        elif isinstance(value, str):
            return [value]
        else:
            raise ValueError("Input must be a list or a string.")

    dirs = convert_to_list(dirs)
    globs = convert_to_list(globs)

    files = []
    for d in dirs:
        path = Path(d).resolve()
        if not path.exists():
            raise ValueError(f'"{path}" doesn\'t exist!')
        for glob in globs:
            matched_files = path.rglob(glob)
            files.extend(matched_files)
    return files


def group_files(files, ngroups):
    file_groups = np.array_split(np.array(files), ngroups)
    return {i: [str(f).strip() for f in group] for i, group in enumerate(file_groups)}


def create_filenames(directory, pattern, ngroups):
    return {i: str(Path(directory) / pattern.format(i)).strip() for i in range(ngroups)}


def create_tasks(ifiles, ofiles, njobs):
    dbase = {}
    for group_id in ifiles:
        task = {"input_files": ifiles[group_id], "output_file": ofiles[group_id]}

        job_id = group_id % njobs
        dbase.setdefault(job_id, {})[group_id] = task

    return dbase


def create_task_db(
    data_dirs,
    data_globs,
    output_dir,
    temp_ngroups,
    temp_njobs,
    final_ngroups,
    final_njobs,
    group_files_fn=group_files,
):

    # Tasks for creating temp files for dst files
    data_files = find_files(data_dirs, data_globs)
    ifiles = group_files_fn(data_files, temp_ngroups)
    ofiles = create_filenames(
        directory=Path(output_dir) / "temp_files",
        pattern="temp_{:05}.h5",
        ngroups=temp_ngroups,
    )
    temp_db = create_tasks(ifiles, ofiles, temp_njobs)

    # Tasks for creating final files from temp files
    temp_files = list(ofiles.values())
    ifiles = group_files(temp_files, final_ngroups)
    ofiles = create_filenames(
        directory=Path(output_dir) / "final_files",
        pattern="final_{:05}.h5",
        ngroups=final_ngroups,
    )
    final_db = create_tasks(ifiles, ofiles, final_njobs)

    return {"temp": temp_db, "final": final_db}
