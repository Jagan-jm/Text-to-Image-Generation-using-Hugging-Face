import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# Load the model (uses GPU if available)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Replace with "cpu" if no GPU available

def generate_image(prompt, style, width, height):
    # Combine style with prompt for creative output
    full_prompt = f"{style}, {prompt}"
    # Stable Diffusion expects multiples of 8 for width and height
    width = (width // 8) * 8
    height = (height // 8) * 8
    image = pipe(full_prompt, width=width, height=height).images[0]
    return image

with gr.Blocks() as demo:
    gr.Markdown("# Text-to-Image Generation with Style and Resolution Control")

    prompt_input = gr.Textbox(label="Prompt", value="A fantasy castle on a mountain at sunset")

    style_dropdown = gr.Dropdown(
        label="Style",
        choices=["realistic", "anime", "cyberpunk", "fantasy", "cartoon"],
        value="fantasy"
    )

    width_slider = gr.Slider(label="Width", minimum=256, maximum=1024, step=64, value=512)
    height_slider = gr.Slider(label="Height", minimum=256, maximum=1024, step=64, value=512)

    output_img = gr.Image(label="Generated Image")

    generate_btn = gr.Button("Generate Image")
    generate_btn.click(
        generate_image,
        inputs=[prompt_input, style_dropdown, width_slider, height_slider],
        outputs=output_img
    )

demo.launch()
