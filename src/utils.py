from diffusers import UNet2DModel
import PIL.Image
import numpy as np
import torch

from diffusers import DDPMScheduler
import tqdm
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from diffusers.utils.remote_utils import remote_decode

from diffusers import AutoencoderKL
import matplotlib.pyplot as plt
import math
import pandas as pd

from torchvision import transforms
import torch
from custom_vae import VAEDog

pipeline = StableDiffusionPipeline.from_pretrained("ckpt/anything-v3.0").to("cuda")

def gen_img(pipeline: StableDiffusionPipeline = pipeline,
            prompt:str='Hatsune Miku -- on the stage',
            vae: AutoencoderKL = None,
            seed:int=44,
            save: bool = False):
      g = torch.Generator(device="cuda")
      if vae is not None:
        pipeline.vae = vae
      g.manual_seed(seed)
      img = pipeline(prompt,
                    generator=g).images[0]
      if save:
        (f"{prompt}.png")
      return img

def plot_images(*images, cols=4):
    n = len(images)
    rows = math.ceil(n / cols)
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def load_vaes(device:str='cuda')-> list:
    vae_paths = {
    "sd_vae_mse": "stabilityai/sd-vae-ft-mse",
    "sd_v1_4_vae": ("CompVis/stable-diffusion-v1-4", "vae"),
    "sdxl_fix_vae": "madebyollin/sdxl-vae-fp16-fix",
    "vae_ft_ema": "stabilityai/sd-vae-ft-ema",
    "waifu_vae": ("hakurei/waifu-diffusion", "vae"),
    "sdxl_official": "stabilityai/sdxl-vae",
    "sdxl_official": "stabilityai/sdxl-vae",
    "sd_vae_ema": "stabilityai/sd-vae-ft-ema",
    "kandinsky_vae": "kandinsky-community/kandinsky-2-2-decoder",
    "sd_v1_5_vae": ("runwayml/stable-diffusion-v1-5", "vae"),
    "small_sd_vae": "segmind/small-sd",
    "flux": "diffusers/FLUX.1-vae",
    "sdxl_vae_fp16": "diffusers/sdxl-vae-fp16-fix",
    "vae_anime": 'ttj/stable-diffusion-vae-anime',
    "ENDMANGA_MIX_VAEFIX": "EnD-Diffusers/ENDMANGA_MIX_VAEFIX",
    "john_doe_vae": 'john-doe-12344/waifu-diffusion-vae',
    "medium_vae": "chendelong/stable-diffusion-3-medium-vae",
    "large_vae": 'huaweilin/stable-diffusion-3.5-large-vae',
    }

    vaes = {}

    for name, path in vae_paths.items():
        print(f"Loading VAE: {name}")
        try:
            if isinstance(path, tuple):
                model_id, subfolder = path
                vae = AutoencoderKL.from_pretrained(model_id, subfolder=subfolder).to(device)
            else:
                vae = AutoencoderKL.from_pretrained(path).to(device)
            vaes[name] = vae
        except Exception as e:
            print(f"Error loading VAE {name}: {e}")
    return vaes


# CKA


def center_gram(gram):
    n = gram.size(0)
    identity = torch.eye(n, device=gram.device)
    ones = torch.ones((n, n), device=gram.device) / n
    return gram - ones @ gram - gram @ ones + ones @ gram @ ones

def compute_cka(X, Y):
    K = X @ X.T
    L = Y @ Y.T

    K_centered = center_gram(K)
    L_centered = center_gram(L)

    numerator = (K_centered * L_centered).sum()
    denominator = torch.norm(K_centered) * torch.norm(L_centered)

    return numerator / denominator

def extract_latents_batch(vae, images, batch_size=8, release_gpu:bool=False):
    all_latents = []
    transform = transforms.ToTensor()
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i+batch_size]
        images_tensor = torch.stack([transform(img).to("cuda") for img in batch_imgs]) * 2 - 1
        with torch.no_grad():
            if isinstance(vae, VAEDog):  # Custom VAE case
                z = vae.encode(images_tensor)
            else:  # diffusers VAE case
                z = vae.encode(images_tensor).latent_dist.sample()
        all_latents.append(z.flatten(1))
    return torch.cat(all_latents, dim=0)

def compare_all_vaes_pandas(vaes: dict, imgs, batch_size=4):
    latent_cache = {}
    names = list(vaes.keys())

    print("Extracting latent vectors...")
    for name in names:
        try:
            z = extract_latents_batch(vaes[name], imgs, batch_size=batch_size)
            latent_cache[name] = z
        except Exception as e:
            print(f"Failed to extract latents for {name}: {e}")
            latent_cache[name] = None

    print("\nComputing CKA matrix...")
    cka_matrix = pd.DataFrame(index=names, columns=names, dtype=float)

    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i <= j:
                z1, z2 = latent_cache[name1], latent_cache[name2]
                if z1 is not None and z2 is not None:
                    try:
                        score = compute_cka(z1, z2).item()
                        cka_matrix.loc[name1, name2] = score
                        cka_matrix.loc[name2, name1] = score  # symetria
                    except Exception as e:
                        print(f"CKA failed for {name1} vs {name2}: {e}")
                        cka_matrix.loc[name1, name2] = None
                        cka_matrix.loc[name2, name1] = None
                else:
                    cka_matrix.loc[name1, name2] = None
                    cka_matrix.loc[name2, name1] = None

    print("\nCKA Matrix:\n")
    print(cka_matrix.round(4))
    return cka_matrix
