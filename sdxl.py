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
    low_vram=False,
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

    # enable VAE titling and slicing if low VRAM
    if low_vram:
        # https://huggingface.co/docs/diffusers/optimization/fp16#tiled-vae-decode-and-encode-for-large-images
        pipe.enable_vae_tiling()
        # https://huggingface.co/docs/diffusers/optimization/fp16#sliced-vae-decode-for-larger-batches
        pipe.enable_vae_slicing()

    # model offloading to save memory:
    # https://huggingface.co/docs/diffusers/optimization/fp16#model_offloading
    if low_vram and device == "cuda":
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
