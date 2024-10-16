import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image
from PIL import Image
import requests
import matplotlib.pyplot as plt

pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16  
)
pipeline.enable_model_cpu_offload() 

url = "https://wallpaperaccess.com/full/2111331.jpg"
init_image = load_image(url)

prompt = "Add a cute cap with soft pastel colors and a small bow on the subject’s head, keeping the overall look natural and adorable."

image = pipeline(prompt=prompt, image=init_image).images[0]

def display_images(init_img, gen_img):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(init_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(gen_img)
    plt.title("Generated Image")
    plt.axis("off")

    plt.show()

display_images(init_image, image)
