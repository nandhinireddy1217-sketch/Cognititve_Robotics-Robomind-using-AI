from backend.voice import listen_to_voice

print("🎤 RoboMind Voice Test")
print("Please speak something...")

result = listen_to_voice()

print("\n🤖 RoboMind heard:")
print(result)