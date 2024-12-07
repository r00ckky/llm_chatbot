import speech_recognition as sr
r=sr.Recognizer()
audio="aud.aiff"
with sr.AudioFile(audio) as source:
    print("Say something")
    audio=r.record(source)
    print("done")
try:
    text=r.recognize_google(audio)
    print(text)
except Exception as e:
    print(e)