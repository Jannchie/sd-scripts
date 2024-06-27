from pathlib import Path


def glob_images_pathlib(dir_path, recursive):
    dir_path = Path(dir_path)
    IMAGE_EXTENSIONS = [
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".PNG",
        ".JPG",
        ".JPEG",
        ".WEBP",
        ".BMP",
    ]
    image_paths = []
    if recursive:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.rglob("*" + ext))
    else:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.glob("*" + ext))
    image_paths = list(set(image_paths))  # 重複を排除
    image_paths.sort()
    return image_paths


def glob_npz_pathlib(dir_path, recursive):
    dir_path = Path(dir_path)
    EXTENSIONS = [".npz"]
    image_paths = []
    if recursive:
        for ext in EXTENSIONS:
            image_paths += list(dir_path.rglob("*" + ext))
    else:
        for ext in EXTENSIONS:
            image_paths += list(dir_path.glob("*" + ext))
    image_paths = list(set(image_paths))  # 重複を排除
    image_paths.sort()
    return image_paths
