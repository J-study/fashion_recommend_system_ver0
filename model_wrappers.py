from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from typing import List

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = None

def init_sd(pipe_path=None, lora_path=None, lora_weight_name=None):
    global pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device=="cuda" else torch.float32,
    ).to(device)

    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()

    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name=lora_weight_name)

    return pipe


def generate_images(init_image: Image.Image, prompt: str,
                    total_images: int = 100, batch_size: int = 10,
                    strength=0.9, guidance_scale=9, num_inference_steps=30) -> List[Image.Image]:

    global pipe
    images_out = []

    for _ in range(total_images // batch_size):
        out = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            negative_prompt="anime, illustration, low quality, cutoff, cropped, nsfw, blurry, bad hands, bad anatomy",
            num_images_per_prompt=batch_size
        ).images

        images_out.extend(out)
        torch.cuda.empty_cache()

    return images_out
