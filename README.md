# Image-to-Image Generation with Stable Diffusion and Kandinsky

This repository contains scripts to generate modified images using different Image-to-Image models like **Stable Diffusion** and **Kandinsky**. Each script loads an initial image and applies the prompt to alter the image using AI-based diffusion techniques.

## `stable_diffusion_img2img.py`

This script uses **Stable Diffusion v1.5** from `runwayml` to modify an image using a textual prompt. It loads an initial image from a URL and applies the given prompt to generate a new image with soft pastel colors and a cute cap with a bow on the subject's head.

### Model:
- **Pipeline**: `StableDiffusionImg2ImgPipeline` from the `runwayml/stable-diffusion-v1-5` model.
- **Image Modification**: Adds soft pastel colors and a cute cap.

### Key Features:
- Enables CPU offloading for memory efficiency.
- Loads the initial image from a URL.

---

## `kandinsky_img2img.py`

This script utilizes the **Kandinsky 2.2 Decoder** to modify an image according to a text prompt. It adds a cute cap and a small bow to the subject in the image while keeping the look natural and adorable.

### Model:
- **Pipeline**: `kandinsky-community/kandinsky-2-2-decoder`.
- **Image Modification**: Adds a cap with soft pastel colors and a bow.

### Key Features:
- Memory-efficient xFormers attention for better performance.
- Uses `enable_model_cpu_offload` for better resource management.

---

## `stable_diffusion_xl_img2img.py`

This script leverages **Stable Diffusion XL Refiner** to apply a prompt-based transformation on an image. The transformation involves adding a cute cap and a bow while maintaining a soft, adorable appearance.

### Model:
- **Pipeline**: `stabilityai/stable-diffusion-xl-refiner-1.0`.
- **Image Modification**: Adds a cap and bow to the subject using refined, high-quality generation.

### Key Features:
- Works with xFormers memory-efficient attention for optimal resource usage.
- Uses CPU offloading for large model support.

---

## Setup and Installation

To get started with any of these scripts, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/shahram8708/Image2Image.git
cd img2img-scripts
```

### 2. Install Dependencies

Install the required libraries using the following commands:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate safetensors Pillow xformers
```

### 3. GPU Support

Ensure you have a compatible GPU for CUDA (if using CUDA) for efficient inference, otherwise these models can be run on the CPU.

---

## Usage

You can run each script individually, depending on the model you want to use. Each script takes an initial image from a URL and applies a prompt to modify the image.

### Example Usage:

1. **Run `stable_diffusion_img2img.py`**:
    ```bash
    python stable_diffusion_img2img.py
    ```

2. **Run `kandinsky_img2img.py`**:
    ```bash
    python kandinsky_img2img.py
    ```

3. **Run `stable_diffusion_xl_img2img.py`**:
    ```bash
    python stable_diffusion_xl_img2img.py
    ```

### Prompt Example:
The scripts use a default prompt to modify the subject:
```
"Add a cute cap with soft pastel colors and a small bow on the subject’s head, keeping the overall look natural and adorable."
```
