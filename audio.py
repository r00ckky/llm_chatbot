import pyaudio
import speech_recognition as sr
def audi(afp):
    recog= sr.Recognizer()
    with sr.AudioFile(afp) as s1:
        ad= recog.record(s1)
    try:
        t=recog.recognize_google(ad)
        return t
    except sr.UnknownValueError:
        return "could not understand"
    except sr.RequestError as e:
        return f"Error making the request; {e}"
result= audi('testaudio.aiff')
print(result)