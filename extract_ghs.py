import os
from pathlib import Path
from rich.progress import Progress

ds_path = "/mnt/nfs-mnj-hot-09/tmp/aibooru_full"

ds_path = Path(ds_path)
tar_paths = list(ds_path.glob("images/*.tar"))
# 对每个 tar 文件进行解压，放到 data 目录下

os.makedirs(ds_path / "data", exist_ok=True)
with Progress() as progress:
    task = progress.add_task("[green]Extracting...", total=len(tar_paths))

    for tar_path in tar_paths:
        progress.console.print(f"Extracting {tar_path}...")
        os.system(f"tar -xf {tar_path} -C {ds_path / 'data'}")
        progress.advance(task)

progress.console.print("Done.")
