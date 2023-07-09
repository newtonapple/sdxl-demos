import utils
import sys

print(sys.argv)

prompt = sys.argv[1]
outfile = sys.argv[2]
pipe = utils.pipeline()

image = pipe(prompt=prompt).images[0]
image.save(outfile)
