import utils
import gradio as gr


class WebUI:
    def __init__(self):
        self.pipeline = utils.pipeline()

        inputs = [
            gr.Textbox(),  # prompt
            gr.Textbox(),  # negative prompt
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
        width,
        height,
        steps,
        guidance_scale,
        num_images_per_prompt,
    ):
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
        )
        # output = self.pipeline(prompt=prompt)
        return output.images

    def launch(self, *args, **kwargs):
        self.webui.launch(*args, **kwargs)


if __name__ == "__main__":
    WebUI().launch()
