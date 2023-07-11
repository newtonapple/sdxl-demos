import torch
from diffusers import DiffusionPipeline


def torch_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def pipeline(
    model="stabilityai/stable-diffusion-xl-base-0.9",
    device=torch_device(),
    watermark=False,
):
    torch_dtype = torch.float16
    variant = "fp16"

    # MacOS can only use fp32
    if device == "mps":
        torch_dtype = torch.float32
        variant = "fp32"

    pipe = DiffusionPipeline.from_pretrained(
        model,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant=variant,
    )

    if device == "cpu":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    # 20-30% inference speed up for torch >= 2.0
    pipe.unit = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    # mock out watermark, as it introduces undesirable pixels:
    # https://github.com/huggingface/diffusers/issues/4014
    if not watermark:
        pipe.watermark = NoWatermark()

    return pipe


class NoWatermark:
    def apply_watermark(self, img):
        return img
