import whisper
import pyaudio
import wave
import time

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

    # Save the audio file with a unique name
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    audFile = f"audios/audiofile_{timestamp}.wav"
    waveFile = wave.open(audFile, 'wb')
    waveFile.setnchannels(1)
    waveFile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    waveFile.setframerate(44100)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    return audFile # return the name of the file

#Transcribing The Audio
def TranscribeAud(Aud) :

    model = whisper.load_model("base")
    result = model.transcribe(Aud, fp16 = False)

    return result["text"]
