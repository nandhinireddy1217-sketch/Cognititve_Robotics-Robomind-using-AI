import ollama

image_path = r"C:\Users\nandh\OneDrive\Pictures\shinchan.jpg"

response = ollama.chat(
    model="gemma3:4b",
    messages=[
        {
            "role": "user",
            "content": "Describe what you see in this image.",
            "images": [image_path]
        }
    ]
)

print("\n🤖 Gemma 3 Vision:")
print(response["message"]["content"])