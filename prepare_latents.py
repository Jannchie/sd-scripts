import json
import os
import cv2
import numpy as np
import torch
from dataclasses import dataclass
from PIL import Image
from typing import Tuple
from diffusers import AutoencoderKL
from torchvision import transforms
from pathlib import Path
import threading
from queue import Queue
from rich.progress import Progress

from utils import glob_images_pathlib


@dataclass
class Args:
    model_name_or_path: str
    mixed_precision: str


class ImageProcessor:
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.vae = self.load_vae_model()
        self.predefined_resos = [
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
        self.predefined_ars = np.array([w / h for w, h in self.predefined_resos])

    def load_vae_model(self):
        vae = (
            AutoencoderKL.from_pretrained(
                self.model_name_or_path, torch_dtype=torch.float32
            )
            .to(self.device)
            .eval()
        )
        return vae

    def select_reso(
        self, image_width: int, image_height: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        aspect_ratio = image_width / image_height
        ar_errors = self.predefined_ars - aspect_ratio
        predefined_bucket_id = np.abs(ar_errors).argmin()
        reso = self.predefined_resos[predefined_bucket_id]

        scale = (
            reso[1] / image_height
            if aspect_ratio > reso[0] / reso[1]
            else reso[0] / image_width
        )
        resized_size = (int(image_width * scale + 0.5), int(image_height * scale + 0.5))
        return reso, resized_size

    def get_crop_ltrb(
        self, bucket_reso: Tuple[int, int], image_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        bucket_ar, image_ar = (
            bucket_reso[0] / bucket_reso[1],
            image_size[0] / image_size[1],
        )
        resized_width, resized_height = (
            (bucket_reso[1] * image_ar, bucket_reso[1])
            if bucket_ar > image_ar
            else (bucket_reso[0], bucket_reso[0] / image_ar)
        )
        crop_left, crop_top = (bucket_reso[0] - int(resized_width)) // 2, (
            bucket_reso[1] - int(resized_height)
        ) // 2
        return (
            crop_left,
            crop_top,
            crop_left + int(resized_width),
            crop_top + int(resized_height),
        )

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        image = Image.open(image_path, "r").convert("RGB")
        return np.array(image)

    def resize_and_trim_image(
        self, image_np: np.ndarray, reso: Tuple[int, int], resized_size: Tuple[int, int]
    ) -> np.ndarray:
        image_np = cv2.resize(image_np, resized_size, interpolation=cv2.INTER_AREA)

        image_height, image_width = image_np.shape[:2]
        if image_width > reso[0]:
            trim_pos = (image_width - reso[0]) // 2
            image_np = image_np[:, trim_pos : trim_pos + reso[0]]
        if image_height > reso[1]:
            trim_pos = (image_height - reso[1]) // 2
            image_np = image_np[trim_pos : trim_pos + reso[1]]
        return image_np

    @torch.no_grad()
    def prepare_image_tensor(self, image_np: np.ndarray) -> torch.Tensor:
        np_to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        image_tensor = np_to_tensor(image_np)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(torch.float32).to(self.device)
        return image_tensor

    @torch.no_grad()
    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        latents = self.vae.encode(image_tensor).latent_dist.sample()[0]
        return latents

    @staticmethod
    def save_encoded_image(
        latents: torch.Tensor,
        crop_ltrb: Tuple[int, int, int, int],
        original_size: Tuple[int, int],
        train_reso: Tuple[int, int],
        save_path: str,
    ):
        new_npz = {
            "latents": latents.cpu().detach().numpy(),
            "crop_ltrb": crop_ltrb,
            "original_size": original_size,
            "train_resolution": train_reso,
        }
        np.savez_compressed(save_path, **new_npz)

    @torch.no_grad()
    def process_image(self, image_path: str, save_path: str):
        image_np = ImageProcessor.load_image(image_path)
        original_size = image_np.shape[1], image_np.shape[0]
        reso, resized_size = self.select_reso(*original_size)
        image_np = self.resize_and_trim_image(image_np, reso, resized_size)
        crop_ltrb = self.get_crop_ltrb(np.array(reso), original_size)
        image_tensor = self.prepare_image_tensor(image_np)
        latents = self.encode_image(image_tensor)
        ImageProcessor.save_encoded_image(
            latents, crop_ltrb, original_size, reso, save_path
        )

    def process_image_and_save_img(self, image_path: str, save_path: str):
        image_np = ImageProcessor.load_image(image_path)
        original_size = image_np.shape[1], image_np.shape[0]
        reso, resized_size = self.select_reso(*original_size)
        image_np = self.resize_and_trim_image(image_np, reso, resized_size)
        Image.fromarray(image_np).save(save_path)

def read_image(
    read_queue, process_queue, lock, progress_counter, path_to_train_resolutions
):
    skip_existing = True
    while True:
        image_path = read_queue.get()
        if image_path is None:
            break
        # 转成 npz 文件的路径
        npz_save_path = image_path.with_suffix(".npz")
        if skip_existing and npz_save_path.exists():
            try:
                npz = np.load(npz_save_path)
                reso = npz.get("train_resolution")
                if reso is not None:
                    # TypeError: Object of type int64 is not JSON serializable
                    reso = reso.tolist()
                    with lock:  # 使用锁来保证线程安全
                        progress_counter[0] += 1
                        path_to_train_resolutions[image_path] = reso
                    continue
            except Exception as e:
                # 读取失败，重新处理
                print(f"Error: {e}")
        try:
            image_np = ImageProcessor.load_image(image_path)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Failed to load {image_path}")
            with lock:
                progress_counter[0] += 1
            continue
        original_size = image_np.shape[1], image_np.shape[0]
        process_queue.put((image_path, image_np, original_size))


@torch.no_grad()
def process_image(process_queue, write_queue, processor, lock, progress_counter):
    while True:
        task = process_queue.get()
        if task is None:
            break  # 退出信号
        image_path, image_np, original_size = task
        reso, resized_size = processor.select_reso(*original_size)
        image_np = processor.resize_and_trim_image(image_np, reso, resized_size)
        crop_ltrb = processor.get_crop_ltrb(np.array(reso), original_size)
        image_tensor = processor.prepare_image_tensor(image_np)
        try:
            latents = processor.encode_image(image_tensor)
            write_queue.put(
                (image_path, latents, crop_ltrb, original_size, np.array(reso))
            )
        except Exception as e:
            print(f"Error: {e}, image path: {image_path}")
            print(image_tensor.shape, reso)
            with lock:
                progress_counter[0]
        # torch.cuda.empty_cache()


def write_image(write_queue, progress_counter, lock, path_to_train_resolutions):
    while True:
        data = write_queue.get()
        if data is None:
            break  # 退出信号
        image_path, latents, crop_ltrb, original_size, reso = data
        npz_save_path = image_path.with_suffix(".npz")
        ImageProcessor.save_encoded_image(
            latents, crop_ltrb, original_size, reso, npz_save_path
        )
        with lock:  # 使用锁来保证线程安全
            progress_counter[0] += 1
            path_to_train_resolutions[image_path] = reso.tolist()


class ImageProcessingPipeline:
    def __init__(self, model_name, path_list, meta_path, num_reader=12, num_writer=12):
        self.meta_path = meta_path
        self.num_reader = num_reader
        self.num_writer = num_writer
        self.path_to_train_resolutions = {}

        self.reader_threads = []
        self.process_threads = []
        self.writer_threads = []

        self.read_queue = Queue(maxsize=num_reader)
        self.process_queue = Queue(maxsize=10)
        self.write_queue = Queue(maxsize=num_writer)

        self.progress_counter = [0]
        self.progress_lock = threading.Lock()

        self.gpu_count = torch.cuda.device_count()
        self.processor_list = [
            ImageProcessor(model_name_or_path=model_name, device=f"cuda:{i}")
            for i in range(self.gpu_count)
        ]
        self.path_list = [Path(p) for p in path_list]

    def setup_threads(self):
        self.reader_threads = [
            threading.Thread(
                target=read_image,
                args=(
                    self.read_queue,
                    self.process_queue,
                    self.progress_lock,
                    self.progress_counter,
                    self.path_to_train_resolutions,
                ),
            )
            for _ in range(self.num_reader)
        ]

        self.process_threads = [
            threading.Thread(
                target=process_image,
                args=(
                    self.process_queue,
                    self.write_queue,
                    processor,
                    self.progress_lock,
                    self.progress_counter,
                ),
            )
            for processor in self.processor_list
        ]

        self.writer_threads = [
            threading.Thread(
                target=write_image,
                args=(
                    self.write_queue,
                    self.progress_counter,
                    self.progress_lock,
                    self.path_to_train_resolutions,
                ),
            )
            for _ in range(self.num_writer)
        ]

    def start_threads(self):
        for t in self.reader_threads:
            t.start()
        for t in self.process_threads:
            t.start()
        for t in self.writer_threads:
            t.start()

    def join_threads(self):
        for t in self.reader_threads:
            t.join()
        for t in self.process_threads:
            t.join()
        for t in self.writer_threads:
            t.join()

    def process_metadata(self):
        if os.path.exists(self.meta_path):
            metadata = json.load(open(self.meta_path, "r"))
        else:
            metadata = {}
        for path, reso in self.path_to_train_resolutions.items():
            if metadata.get(str(path)) is None:
                metadata[str(path)] = {}
            metadata[str(path)]["train_resolution"] = reso
        # save metadata
        json.dump(metadata, open(self.meta_path, "w"), indent=2)

    def run(self):
        self.setup_threads()
        self.start_threads()

        total_processes = len(self.path_list)

        with Progress() as progress:
            task = progress.add_task("Processing images...", total=total_processes)

            for path in self.path_list:
                self.read_queue.put(path)
                progress.update(task, advance=1)

            while self.progress_counter[0] < total_processes:
                with self.progress_lock:
                    progress.update(task, completed=self.progress_counter[0])

        for _ in range(self.num_writer):
            self.write_queue.put(None)
        for _ in range(self.num_reader):
            self.read_queue.put(None)
        for _ in range(len(self.processor_list)):
            self.process_queue.put(None)

        self.join_threads()
        self.process_metadata()


if __name__ == "__main__":

    ds_path = "/mnt/nfs-mnj-hot-09/tmp/aibooru_full/data"
    meta_path = "/mnt/nfs-mnj-hot-09/tmp/aibooru_full/meta.json"
    model_name = "madebyollin/sdxl-vae-fp16-fix"
    # path_list = glob_images_pathlib(ds_path, True)
    # pipeline = ImageProcessingPipeline(model_name, path_list, meta_path)
    # pipeline.run()

    processor = ImageProcessor(model_name_or_path=model_name, device=f"cuda")
    processor.process_image_and_save_img('/mnt/nfs-mnj-hot-09/tmp/yande/040/1043040.png', '1.png')