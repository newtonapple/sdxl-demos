import utils
import sys

if __name__ == "__main__":
    prompt = sys.argv[1]
    outfile = sys.argv[2]
    pipe = utils.pipeline()

    image = pipe(prompt=prompt).images[0]
    image.save(outfile)
