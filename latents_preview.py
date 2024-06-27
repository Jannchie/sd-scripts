# %%
import json
import random

import torch

meta_path = "/mnt/nfs-mnj-hot-09/tmp/yande/meta.json"

metadata = json.load(open(meta_path, encoding="utf-8"))
keys = list(metadata.keys())
keys[15]
# 检查一个方形的图像
# %%
from pathlib import Path

# 过滤key，需要 train_resolution 是 1024x1024 的key
keys = [k for k in keys if metadata[k]["train_resolution"] == [1024, 1024]]
# get random key
key = random.choice(keys)
print(key)
npz_path = Path(key).with_suffix(".npz")
npz_path

import numpy as np

npz = np.load(npz_path)
# %%
from diffusers import AutoencoderKL

vae = (
    AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    .to("cuda")
    .eval()
)

# %%
import gc

torch.cuda.empty_cache()
gc.collect()
sample = np.array([npz["latents"]])
img_array = (
    vae.decode(torch.tensor(sample, dtype=torch.float16).to("cuda"))
    .sample.detach()
    .cpu()
    .numpy()[0]
)
img_array = np.clip(img_array / 2 + 0.5, 0, 1)
img_array = (img_array * 255).astype(np.uint8)
# channel, width, height => width, height, channel
img_array = np.moveaxis(img_array, 0, -1)
import PIL

PIL.Image.fromarray(img_array).save("1.png")
# %%
