import torch
from diffusers import DiffusionPipeline
import numpy as np

# Load Stable Diffusion XL Base1.0
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

# Optional CPU offloading to save some GPU Memory
pipe.enable_model_cpu_offload()

# Loading Trained LoRA Weights
pipe.load_lora_weights("AdamLucek/sdxl-base-1.0-oldbookillustrations-lora")

def generate_image(prompt, height, width, num_inference_steps, guidance_scale):
    result = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
    )
    image = result.images[0]
    return image

def convert_2d_to_3d(image):
    # Convert 2D image to 3D model
    # This is a simplified example and may require more complex logic
    # to create a 3D model from a 2D image
    height, width, channels = image.shape
    vertices = np.zeros((height * width * 3, 3))
    for i in range(height):
        for j in range(width):
            vertices[i * width * 3 + j * 3] = j / width
            vertices[i * width * 3 + j * 3 + 1] = i / height
            vertices[i * width * 3 + j * 3 + 2] = 0
    return vertices

def process_request(prompt, height, width, num_inference_steps, guidance_scale):
    image = generate_image(prompt, height, width, num_inference_steps, guidance_scale)
    vertices = convert_2d_to_3d(image)
    return vertices

def generate_3d_model(prompt, height, width, num_inference_steps, guidance_scale):
    vertices = process_request(prompt, height, width, num_inference_steps, guidance_scale)
    return vertices

def generate_2d_image(prompt, height, width, num_inference_steps, guidance_scale):
    image = generate_image(prompt, height, width, num_inference_steps, guidance_scale)
    return image

if __name__ == "__main__":
    prompt = "Room with 3 doors and 1 sofa"
    height = 1024
    width = 1024
    num_inference_steps = 50
    guidance_scale = 7.0

    vertices = generate_3d_model(prompt, height, width, num_inference_steps, guidance_scale)
    print(vertices)

    image = generate_2d_image(prompt, height, width, num_inference_steps, guidance_scale)
    print(image.shape)