from backend.vision_ai import analyze_image


image_path = r"C:\Users\nandh\OneDrive\Pictures\shinchan.jpg"

result = analyze_image(image_path)

print("\n🤖 RoboMind Vision:")
print(result)
