import torch
import random
import gradio as gr
from sdxl import pipeline


class WebUI:
    def __init__(self):
        self.pipeline = pipeline(model="stabilityai/stable-diffusion-xl-base-0.9")
        self.refiner = pipeline(model="stabilityai/stable-diffusion-xl-refiner-0.9")

        inputs = [
            gr.Textbox(),  # prompt
            gr.Textbox(),  # negative prompt
            gr.Textbox(value=-1),  # seed
            gr.Slider(5, 2000, value=1024, step=8),  # width
            gr.Slider(5, 2000, value=1024, step=8),  # height
            gr.Slider(0, 200, value=50, step=1),  # steps
            gr.Slider(0, 30, value=7.5),  # scale
            gr.Slider(1, 30, value=1, step=1),  # num of images
        ]
        self.webui = gr.Interface(self.text_to_img, inputs, gr.Gallery())

    def text_to_img(
        self,
        prompt,
        negative_prompt,
        seed,
        width,
        height,
        steps,
        guidance_scale,
        num_images_per_prompt,
    ):
        seed = int(seed)
        if seed <= -1:
            seeds = [
                random.randint(1, 99999999999) for _ in range(0, num_images_per_prompt)
            ]
            generator = [
                torch.Generator(device=self.pipeline.device).manual_seed(s)
                for s in seeds
            ]
        else:
            seeds = [seed + i for i in range(0, num_images_per_prompt)]
            generator = [
                torch.Generator(device=self.pipeline.device).manual_seed(s)
                for s in seeds
            ]

        latent_image = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            output_type="latent",
        ).images

        images = self.refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            image=latent_image,
        ).images

        return [[i[1], seeds[i[0]]] for i in enumerate(images)]

    def launch(self, *args, **kwargs):
        self.webui.launch(*args, **kwargs)


if __name__ == "__main__":
    WebUI().launch()
