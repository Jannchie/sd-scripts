from pathlib import Path
import torch
from wdtagger import Tagger, Result
from PIL import Image
from rich.progress import Progress
from multiprocessing import Process, Queue, cpu_count
import os


import argparse


def worker(task_queue, progress_queue, i):
    """
    Worker function to process images using Tagger.

    Args:
        task_queue (Queue): Queue containing lists of image file paths to process.
        progress_queue (Queue): Queue to update progress information.
        i (int): Device ID for the worker.
    """

    tagger = Tagger(
        loglevel="ERROR",
        num_threads=1,
        providers=[
            ("CUDAExecutionProvider", {"device_id": i}),
            "CPUExecutionProvider",
        ],
    )

    while True:
        batch_files = task_queue.get()
        if batch_files is None:
            break  # terminate signal received
        len_before = len(batch_files)
        batch_files = [
            f for f in batch_files if not Path(f).with_suffix(".txt").exists()
        ]

        results = []
        if batch_files:
            try:
                images = [Image.open(image_path) for image_path in batch_files]
                results = tagger.tag(images)
            except Exception:
                continue
            # if result is not iterable, make it iterable
            if not isinstance(results, list):
                results = [results]
            for image_path, result in zip(batch_files, results):
                tags_path = Path(image_path).with_suffix(".txt")
                assert isinstance(result, Result)
                tags_path.write_text(result.general_tags_string, encoding="utf-8")

        progress_queue.put(len_before)

    # notify main process this worker has done its job
    progress_queue.put(None)


def main():
    target_dir = Path("/mnt/nfs-mnj-hot-09/tmp/yande/")
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
    files = [str(f) for f in target_dir.glob("**/*") if f.suffix in IMAGE_EXTENSIONS]
    batch_size = 32

    task_queue = Queue()
    progress_queue = Queue()
    for i in range(0, len(files), batch_size):
        batch_files = files[i : i + batch_size]
        task_queue.put(batch_files)

    num_workers = 2  # Adjust based on available CPUs
    gpu_count = torch.cuda.device_count()
    workers = [
        Process(target=worker, args=(task_queue, progress_queue, i % gpu_count))
        for i in range(num_workers)
    ]

    for w in workers:
        w.start()

    # Send a termination signal (None) to each worker after tasks
    for _ in workers:
        task_queue.put(None)

    with Progress() as progress:
        task = progress.add_task(f"Processing images...", total=len(files))

        processed_count = 0
        done_workers = 0

        while done_workers < num_workers:
            item = progress_queue.get()
            if item is None:
                done_workers += 1
            else:
                processed_count += item
                remaining_count = len(files) - processed_count
                progress.update(
                    task,
                    completed=processed_count,
                    description=f"Processing images... {processed_count}/{len(files)}, {remaining_count} remaining",
                )

        for w in workers:
            w.join()


if __name__ == "__main__":
    main()
