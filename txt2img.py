import argparse
from sdxl import pipeline


def arguments_parser():
    parser = argparse.ArgumentParser(
        description="Generate a 1024x1024 image for the given prompt to the specified output file."
    )
    parser.add_argument(
        "prompt",
        type=str,
        help='prompt for the image (e.g., "sunshrine and a blue sky")',
    )
    parser.add_argument(
        "filename",
        type=str,
        help="filename of the output image (e.g., images/sky.png)",
    )
    parser.add_argument(
        "-l",
        "--lowvram",
        default=False,
        action="store_true",
        help="enable lowvram mode.",
    )
    return parser


def main():
    args = arguments_parser().parse_args()
    prompt = args.prompt
    outfile = args.filename
    low_vram = args.lowvram

    base_pipe = pipeline(
        model="stabilityai/stable-diffusion-xl-base-0.9", low_vram=low_vram
    )
    refiner_pipe = pipeline(
        model="stabilityai/stable-diffusion-xl-refiner-0.9", low_vram=low_vram
    )

    images = base_pipe(prompt=prompt, output_type="latent").images
    image = refiner_pipe(prompt=prompt, image=images).images[0]
    image.save(outfile)


if __name__ == "__main__":
    main()
