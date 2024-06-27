# %%
import pandas as pd


# load csv
csv_path_str = (
    "/mnt/nfs-mnj-archive-12/group/creative/pan/datasets/aibooru_full/table.csv"
)
df = pd.read_csv(csv_path_str)
df
#set index to filename
df.set_index("filename", inplace=True)
# order by views desc
#%%
import json
from rich.progress import Progress
from pathlib import Path
meta_path_str = "/mnt/nfs-mnj-hot-09/tmp/aibooru_full/meta.json"
metadata = json.load(open(meta_path_str, "r", encoding="utf-8"))
with Progress() as progress:
    task = progress.add_task("[cyan]Processing metadata...", total=len(metadata))

    for key in metadata.keys():
        filename = Path(key).name
        tags = df['tags'][filename]
        metadata[key]["tags"] = tags
        progress.update(task, advance=1)

with open(meta_path_str, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

# python fintune/merge_all_to_metadata.py