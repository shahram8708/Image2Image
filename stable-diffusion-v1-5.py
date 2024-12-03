import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.utils import load_image

pipeline = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-full-face_sd15.bin")

pipeline.set_ip_adapter_scale(0.5)

def generate_image(prompt, ip_adapter_image, negative_prompt=""):
    ip_image = load_image(ip_adapter_image)
    
    generator = torch.Generator(device="cuda").manual_seed(26)
    
    num_inference_steps = 100
    
    output = pipeline(
        prompt=prompt,
        ip_adapter_image=ip_image,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator
    ).images[0]
    
    return output

with gr.Blocks() as demo:
    gr.Markdown(
        """
        ## AI Image Generation with IP Adapter
        Create high-quality images with customized prompts using the Stable Diffusion model enhanced with an IP Adapter. 
        Upload an image to use as input and modify your prompt to control the output. This tool is designed for a professional and creative touch.
        """
    )

    ip_image_input = gr.File(label="Upload Image for IP Adapter", type="filepath")
    
    prompt_input = gr.Textbox(
        label="Enter Your Prompt",
        placeholder="Describe the image you want to generate"
    )

    negative_prompt_input = gr.Textbox(
        label="Enter Negative Prompt (Optional)",
        placeholder="Specify what to avoid in the image"
    )

    generate_button = gr.Button("Generate Image")
    
    output_image = gr.Image(label="Generated Image")

    generate_button.click(
        generate_image,
        inputs=[prompt_input, ip_image_input, negative_prompt_input],
        outputs=output_image
    )

demo.launch(share=True)
