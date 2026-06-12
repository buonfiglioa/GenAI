###############################################
###       BEFORE RUNNING THIS SCRIPT        ###
### MAKE SURE TO START THE LOCAL LLM SERVER ###
###############################################
# 1 - Open LM Studio
# 2 - Download the gemma-3-4b model
# 3 - Go to the "Local Server" panel
# 4 - Load the gemma model
# 5 - Start the server toggling the button in the "status" box, on the upper left side
# 6 - Run this script to test the connection and the response of the model


from openai import OpenAI

#############################
### Text-only interaction ###
#############################

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

response = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print("\nTEXT-ONLY INTERACTION - START")
print(response.choices[0].message.content)
print("TEXT-ONLY INTERACTION - END\n")



#############################################
### Multimodal interaction (text + image) ###
#############################################

import base64
import requests
from PIL import Image
from io import BytesIO

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

image_url = "https://www.daisho.be/wp-content/uploads/2021/08/style-karate-ecole.jpg"
image_response = requests.get(image_url)
image = Image.open(BytesIO(image_response.content))
buffered = BytesIO()
image.save(buffered, format="JPEG")
image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

response = client.chat.completions.create(
    model="gemma-3-4b",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }
    ],
    max_tokens=500
)

print("\nMULTIMODAL INTERACTION - START")
print(response.choices[0].message.content)
print("MULTIMODAL INTERACTION - END\n")