import numpy as np
from rich.progress import Progress
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import glob_npz_pathlib


def process_file(path):
    try:
        # load npz
        npz = np.load(path)
        assert npz["train_resolution"][0] == npz["latents"].shape[2] * 8
        assert npz["train_resolution"][1] == npz["latents"].shape[1] * 8
    except Exception:
        print(f"Failed: {path}")
        return path
    return None


if __name__ == "__main__":
    ds_path = "/mnt/nfs-mnj-hot-09/tmp/yande"
    path_list = glob_npz_pathlib(ds_path, True)

    failed_paths = []

    with Progress() as progress:
        task = progress.add_task("Processing files...", total=len(path_list))

        with ThreadPoolExecutor() as executor:
            future_to_path = {
                executor.submit(process_file, path): path for path in path_list
            }

            for future in as_completed(future_to_path):
                path = future_to_path[future]
                result = future.result()
                if result:
                    failed_paths.append(result)
                progress.advance(task)

    # Output all failed paths
    if failed_paths:
        print("\nFailed paths:")
        for path in failed_paths:
            print(path)
    else:
        print("All files passed.")
