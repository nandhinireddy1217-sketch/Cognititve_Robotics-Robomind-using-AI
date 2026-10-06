import speech_recognition as sr
import pyttsx3


def listen_to_voice():

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:

        print("🎤 Listening...")

        recognizer.adjust_for_ambient_noise(
            source,
            duration=1
        )

        audio = recognizer.listen(source)

    try:

        text = recognizer.recognize_google(
            audio
        )

        return text

    except sr.UnknownValueError:

        return "Sorry, I could not understand your voice."

    except sr.RequestError:

        return "Speech recognition service is unavailable."


def speak_text(text):

    engine = pyttsx3.init()

    engine.say(text)

    engine.runAndWait()