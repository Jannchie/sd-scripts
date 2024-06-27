import json
from rich.progress import Progress
from wdtagger import Tagger
from PIL import Image

tagger = Tagger(num_threads=1)


def check_image_paths_with_progress(json_file):
    # 读取JSON文件
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # 创建一个进度条
    invalid_paths = []
    total_paths = len(data.keys())

    with Progress() as progress:
        task = progress.add_task("[green]Checking images...", total=total_paths)
        for image_path in data.keys():
            # 如果不存在 tags 或 caption 字段，说明这个图片路径是无效的
            if "tags" not in data[image_path]:
                try:
                    result = tagger.tag(Image.open(image_path))
                except Exception as e:
                    print(f"Error: {e}")
                    invalid_paths.append(image_path)
                    continue
                tags = result.general_tags_string
                data[image_path]["tags"] = tags
                print(f"path: {image_path}, tags: {tags}")
            progress.update(task, advance=1)
        print("invalid_paths count:", len(invalid_paths))
        for path in invalid_paths:
            # del
            data.pop(path)
    # save jsonfile
    with open(json_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    return


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("json_file_path", type=str)
if __name__ == "__main__":
    args = parser.parse_args()
    json_file_path = args.json_file_path
    check_image_paths_with_progress(json_file_path)
