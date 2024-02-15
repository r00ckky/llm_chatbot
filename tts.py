import pyttsx3
from datetime import datetime


def text_to_speech(text):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate-50)
    engine.save_to_file(text, f'output_{datetime.now().strftime("%Y%m%d-%H%M%S")}.mp3')
    engine.runAndWait()

if __name__ == "__main__" :
    text_to_speech(input('Enter Text : '))