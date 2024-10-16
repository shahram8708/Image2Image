import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid
import requests
from PIL import Image
import matplotlib.pyplot as plt

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()

pipeline.enable_xformers_memory_efficient_attention()

url = "https://wallpaperaccess.com/full/2111331.jpg"
init_image = Image.open(requests.get(url, stream=True).raw)

prompt = "Add a cute cap with soft pastel colors and a small bow on the subject’s head, keeping the overall look natural and adorable."

result_image = pipeline(prompt, image=init_image, strength=0.5).images[0]

def display_images(init_img, gen_img):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(init_img)
    ax[0].set_title('Initial Image')
    ax[0].axis('off')

    ax[1].imshow(gen_img)
    ax[1].set_title('Generated Image')
    ax[1].axis('off')

    plt.show()

display_images(init_image, result_image)
