import whisper
import pyaudio
import wave

#capturing Audio till keyboard interrupts
def CapAudio() :

    audio = pyaudio.PyAudio()
    stream = audio.open(format = pyaudio.paInt16, channels=1, rate=44100, input = True, frames_per_buffer = 1024)

    frames = []

    try :
        while True:
            data = stream.read(1024)
            frames.append(data)

    except KeyboardInterrupt:
        pass

    stream.stop_stream()
    stream.close()
    audio.terminate()

    #Creating The Audio File
    sound_file = wave.open("myaud.wav","wb")
    sound_file.setnchannels(1)
    sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    sound_file.setframerate(44100)

    sound_file.writeframes(b''.join(frames))
    sound_file.close()

    #File stored in a var
    audFile = 'myaud.wav'

#Transcribing The Audio
def TranscribeAud(Aud) :

    model = whisper.load_model("base")
    result = model.transcribe(Aud, fp16 = False)

    return result["text"]