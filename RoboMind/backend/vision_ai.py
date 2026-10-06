import ollama


def analyze_image(image_path):
    response = ollama.chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "user",
                "content": "Describe what you see in this image clearly and simply.",
                "images": [image_path]
            }
        ]
    )

    return response["message"]["content"]
