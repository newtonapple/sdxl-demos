import sys
from sdxl import pipeline

if __name__ == "__main__":
    prompt = sys.argv[1]
    outfile = sys.argv[2]
    base_pipe = pipeline(model="stabilityai/stable-diffusion-xl-base-0.9")
    refiner_pipe = pipeline(model="stabilityai/stable-diffusion-xl-refiner-0.9")

    images = base_pipe(prompt=prompt, output_type="latent").images
    image = refiner_pipe(prompt=prompt, image=images).images[0]
    image.save(outfile)
