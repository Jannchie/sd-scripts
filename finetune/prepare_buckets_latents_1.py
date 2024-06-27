import multiprocessing as mp

# 设置 start method
mp.set_start_method("spawn", force=True)

import argparse
import os
import json
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from library.device_utils import init_ipex

init_ipex()

from torchvision import transforms

import library.model_util as model_util
import library.train_util as train_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

gpu_num = torch.cuda.device_count()

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def get_npz_filename(data_dir, image_key, is_full_path, recursive):
    if is_full_path:
        base_name = os.path.splitext(os.path.basename(image_key))[0]
        relative_path = os.path.relpath(os.path.dirname(image_key), data_dir)
    else:
        base_name = image_key
        relative_path = ""

    if recursive and relative_path:
        return os.path.join(data_dir, relative_path, base_name) + ".npz"
    else:
        return os.path.join(data_dir, base_name) + ".npz"


def process_data(
    args, image_paths, metadata, vae_params, device, thread_id, return_dict
):
    logger.info(f"Process {thread_id} processing data on {device}")

    vae = model_util.load_vae(*vae_params)
    vae.eval()
    vae.to(device)

    # bucketのサイズを計算する
    max_reso = tuple([int(t) for t in args.max_resolution.split(",")])
    assert (
        len(max_reso) == 2
    ), f"illegal resolution (not 'width,height') / 画像サイズに誤りがあります。'幅,高さ'で指定してください: {args.max_resolution}"

    if True:
        bucket_manager = train_util.BucketManager(
            False, None, None, None, None
        )
        # use predefined bucket resos
        bucket_manager.set_predefined_resos(
            [
                (640, 1536),
                (768, 1344),
                (832, 1216),
                (896, 1152),
                (1024, 1024),
                (1152, 896),
                (1216, 832),
                (1344, 768),
                (1536, 640),
            ]
        )
    elif not args.bucket_no_upscale:
        bucket_manager = train_util.BucketManager(
            args.bucket_no_upscale,
            max_reso,
            args.min_bucket_reso,
            args.max_bucket_reso,
            args.bucket_reso_steps,
        )
        bucket_manager.make_buckets()
    else:
        logger.warning(
            "min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, because bucket reso is defined by image size automatically / bucket_no_upscaleが指定された場合は、bucketの解像度は画像サイズから自動計算されるため、min_bucket_resoとmax_bucket_resoは無視されます"
        )
    # 画像をひとつずつ適切なbucketに割り当てながらlatentを計算する
    img_ar_errors = []

    def process_batch(is_last):
        for bucket in bucket_manager.buckets:
            if (is_last and len(bucket) > 0) or len(bucket) >= args.batch_size:
                train_util.cache_batch_latents(
                    vae, True, bucket, args.flip_aug, False, False
                )
                bucket.clear()

    dataset = train_util.ImageLoadingDataset(image_paths)
    data = (
        torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.max_data_loader_n_workers,
            collate_fn=collate_fn_remove_corrupted,
            drop_last=False,
        )
        if args.max_data_loader_n_workers
        else [[(None, ip)] for ip in image_paths]
    )

    bucket_counts = {}
    for data_entry in tqdm(data, smoothing=0.0):
        if data_entry[0] is None:
            continue

        img_tensor, image_path = data_entry[0]
        image = (
            transforms.functional.to_pil_image(img_tensor)
            if img_tensor is not None
            else process_image(image_path)
        )

        if image is None:
            continue

        process_image_metadata(
            image,
            image_path,
            metadata,
            args,
            bucket_manager,
            bucket_counts,
            img_ar_errors,
        )
        process_batch(False)

    process_batch(True)
    bucket_manager.sort()

    for i, reso in enumerate(bucket_manager.resos):
        count = bucket_counts.get(reso, 0)
        if count > 0:
            logger.info(f"Process {thread_id} - bucket {i} {reso}: {count}")
    img_ar_errors_mean = np.mean(img_ar_errors)

    return_dict[thread_id] = (metadata, img_ar_errors_mean, bucket_counts)


def process_image(image_path):
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        logger.error(
            f"Could not load image path / 画像を読み込めません: {image_path}, error: {e}"
        )
        return None


def process_image_metadata(
    image, image_path, metadata, args, bucket_manager, bucket_counts, img_ar_errors
):
    image_key = (
        image_path
        if args.full_path
        else os.path.splitext(os.path.basename(image_path))[0]
    )

    if image_key not in metadata:
        metadata[image_key] = {}

    reso, resized_size, ar_error = bucket_manager.select_bucket(
        image.width, image.height
    )

    img_ar_errors.append(abs(ar_error))
    bucket_counts[reso] = bucket_counts.get(reso, 0) + 1

    metadata[image_key]["train_resolution"] = (
        reso[0] - reso[0] % 8,
        reso[1] - reso[1] % 8,
    )

    if not args.bucket_no_upscale:
        assert_resized_size(reso, resized_size, image)

    npz_file_name = get_npz_filename(
        args.train_data_dir, image_key, args.full_path, args.recursive
    )

    if args.skip_existing and train_util.is_disk_cached_latents_is_expected(
        reso, npz_file_name, args.flip_aug
    ):
        return

    image_info = train_util.ImageInfo(image_key, 1, "", False, image_path)
    image_info.latents_npz = npz_file_name
    image_info.bucket_reso = reso
    image_info.resized_size = resized_size
    image_info.image = image
    bucket_manager.add_image(reso, image_info)


def assert_resized_size(reso, resized_size, image):
    assert (
        resized_size[0] == reso[0] or resized_size[1] == reso[1]
    ), f"Internal error, resized size not match: {reso}, {resized_size}, {image.width}, {image.height}"
    assert (
        resized_size[0] >= reso[0] and resized_size[1] >= reso[1]
    ), f"Internal error, resized size too small: {reso}, {resized_size}, {image.width}, {image.height}"


def main(args):
    if args.bucket_reso_steps % 8 > 0:
        logger.warning(
            f"Resolution of buckets in training time is a multiple of 8 / 学習時の各bucketの解像度は8単位になります"
        )
    if args.bucket_reso_steps % 32 > 0:
        logger.warning(
            f"WARNING: bucket_reso_steps is not divisible by 32. It is not working with SDXL / bucket_reso_stepsが32で割り切れません。SDXLでは動作しません"
        )

    # train_data_dir_path = Path(args.train_data_dir)
    # image_paths: List[str] = [str(p) for p in train_util.glob_images_pathlib(train_data_dir_path, args.recursive)]

    if os.path.exists(args.in_json):
        logger.info(f"Loading existing metadata: {args.in_json}")
        with open(args.in_json, "rt", encoding="utf-8") as f:
            metadata = json.load(f)
            # remove all train_resolution
            for v in metadata.values():
                if "train_resolution" in v:
                    del v["train_resolution"]
    else:
        logger.error(f"No metadata / メタデータファイルがありません: {args.in_json}")
        return

    image_paths = list(metadata.keys())

    logger.info(f"Found {len(image_paths)} images.")

    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        args.mixed_precision, torch.float32
    )

    max_reso = tuple(map(int, args.max_resolution.split(",")))
    assert (
        len(max_reso) == 2
    ), f"Illegal resolution (not 'width,height') / 画像サイズに誤りがあります。'幅,高さ'で指定してください: {args.max_resolution}"

    devices = [
        f"cuda:{i}" for i in range(gpu_num)
    ]  # Modify this list if you have more or fewer than 8 GPUs
    num_processes = 1  # You can set this to higher than the number of GPUs you have
    paths_per_process = len(image_paths) // num_processes

    manager = mp.Manager()
    return_dict = manager.dict()

    processes = []
    for i in range(num_processes):
        device = devices[
            i % len(devices)
        ]  # This ensures processes will loop over available devices
        vae_params = (args.model_name_or_path, weight_dtype)

        start_idx = i * paths_per_process
        end_idx = (
            len(image_paths)
            if i == num_processes - 1
            else start_idx + paths_per_process
        )

        process = mp.Process(
            target=process_data,
            args=(
                args,
                image_paths[start_idx:end_idx],
                metadata,
                vae_params,
                device,
                i,
                return_dict,
            ),
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    aggregated_metadata = {}
    aggregated_ar_errors = []
    aggregated_bucket_counts = {}

    for result in return_dict.values():
        metadata_fragment, ar_error_mean, bucket_counts_fragment = result
        for key, value in metadata_fragment.items():
            if "train_resolution" in value:
                aggregated_metadata[key] = value
        aggregated_ar_errors.append(ar_error_mean)
        for reso, count in bucket_counts_fragment.items():
            aggregated_bucket_counts[reso] = (
                aggregated_bucket_counts.get(reso, 0) + count
            )

    logger.info(f"Writing metadata: {args.out_json}")
    with open(args.out_json, "wt", encoding="utf-8") as f:
        json.dump(aggregated_metadata, f, indent=2)

    logger.info("Bucket counts:")
    for reso, count in aggregated_bucket_counts.items():
        logger.info(f"{reso}: {count}")

    logger.info(f"Mean AR error across all processes: {np.mean(aggregated_ar_errors)}")
    logger.info("Done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_data_dir",
        type=str,
        help="Directory for train images / 学習画像データのディレクトリ",
    )
    parser.add_argument(
        "in_json", type=str, help="Metadata file to input / 読み込むメタデータファイル"
    )
    parser.add_argument(
        "out_json",
        type=str,
        help="Metadata file to output / メタデータファイル書き出し先",
    )
    parser.add_argument(
        "model_name_or_path",
        type=str,
        help="Model name or path to encode latents / latentを取得するためのモデル",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size in inference / 推論時のバッチサイズ",
    )
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=None,
        help="Enable image reading by DataLoader with this number of workers (faster) / DataLoaderによる画像読み込みを有効にしてこのワーカー数を適用する（読み込みを高速化）",
    )
    parser.add_argument(
        "--max_resolution",
        type=str,
        default="1536,1536",
        help="Max resolution in fine tuning (width,height) / fine tuning時の最大画像サイズ 「幅,高さ」（使用メモリ量に関係します）",
    )
    parser.add_argument(
        "--min_bucket_reso",
        type=int,
        default=640,
        help="Minimum resolution for buckets / bucketの最小解像度",
    )
    parser.add_argument(
        "--max_bucket_reso",
        type=int,
        default=1536,
        help="Maximum resolution for buckets / bucketの最大解像度",
    )
    parser.add_argument(
        "--bucket_reso_steps",
        type=int,
        default=64,
        help="Steps of resolution for buckets, divisible by 8 is recommended / bucketの解像度の単位、8で割り切れる値を推奨します",
    )
    parser.add_argument(
        "--bucket_no_upscale",
        action="store_true",
        help="Make bucket for each image without upscaling / 画像を拡大せずbucketを作成します",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Use mixed precision / 混合精度を使う場合、その精度",
    )
    parser.add_argument(
        "--full_path",
        action="store_true",
        help="Use full path as image-key in metadata (supports multiple directories) / メタデータで画像キーをフルパスにする（複数の学習画像ディレクトリに対応）",
    )
    parser.add_argument(
        "--flip_aug",
        action="store_true",
        help="Flip augmentation, save latents for flipped images / 左右反転した画像もlatentを取得、保存する",
    )
    parser.add_argument(
        "--alpha_mask",
        type=str,
        default="",
        help="Save alpha mask for images for loss calculation / 損失計算用に画像のアルファマスクを保存する",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip images if npz already exists (both normal and flipped exists if flip_aug is enabled) / npzが既に存在する画像をスキップする（flip_aug有効時は通常、反転の両方が存在する画像をスキップ）",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively look for training tags in all child folders of train_data_dir / train_data_dirのすべての子フォルダにある学習タグを再帰的に探す",
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
