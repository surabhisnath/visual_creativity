import os
from openai import OpenAI
from PIL import Image
import requests

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# image = Image.open("../stimuli/originals/stimuli_G.png").convert("RGBA")
# mask = Image.open("../stimuli/transparent_mask/stimuli_G.png").convert("RGBA")
# image.save("../stimuli/originals/stimuli_G.png", format="PNG")
# mask.save("../stimuli/transparent_mask/stimuli_G.png", format="PNG")

maxnum = 10
response = client.images.edit(
    image=open("../stimuli/originals/stimuli_G.png", "rb"),
    mask=open("../stimuli/padded_mask/stimuli_G.png", "rb"),
    prompt="a basic object or figure, drawn with digital fineliner sketchpen and simple imperfect children's rough doodle on white background, monochrome",
    n=maxnum,
    size="512x512",
)

urls = [resp.url for resp in response.data]

run = 2
for i, url in enumerate(urls):
    response = requests.get(url)

    if response.status_code == 200:
        # Open a file to write the image
        with open(f"../data/AI/stimuli_G/Dalle{run * 10 + i + 1}.png", "wb") as file:
            file.write(response.content)
    else:
        print(f"Failed to download image. Status code: {response.status_code}")
