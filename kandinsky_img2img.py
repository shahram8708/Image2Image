import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image
import requests
from PIL import Image
import matplotlib.pyplot as plt

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, use_safetensors=True
)
pipeline.enable_model_cpu_offload()

pipeline.enable_xformers_memory_efficient_attention()

url = "https://wallpaperaccess.com/full/2111331.jpg"
init_image = Image.open(requests.get(url, stream=True).raw)

prompt = "Add a cute cap with soft pastel colors and a small bow on the subject’s head, keeping the overall look natural and adorable."

image = pipeline(prompt, image=init_image).images[0]

def display_images(init_img, gen_img):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(init_img)
    ax[0].set_title('Initial Image')
    ax[0].axis('off')

    ax[1].imshow(gen_img)
    ax[1].set_title('Generated Image')
    ax[1].axis('off')

    plt.show()
    
display_images(init_image, image)
