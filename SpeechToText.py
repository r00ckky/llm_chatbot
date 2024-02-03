from scipy.io.wavfile import write
import sounddevice as sd
import soundfile as sf
import time
import threading as th
import numpy as np
import pyaudio
from transformers import pipeline
from pynput import keyboard as key
import queue
import sys
import tempfile

whisper = pipeline(
    'automatic-speech-recognition',
    model = 'openai/whisper-tiny',
    device = 0
)

def on_press(key):
    pass

def on_release(key):
    if key == key.Key.esc:
        return False
    
def sound_device():
    
    listener = key.Listener(on_press=on_press, on_release=on_release)
    q = queue.Queue()
    fs = 44100
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    audFile = tempfile.mktemp(prefix=f"audiofile_{timestamp}", suffix='.wav', dir="audios")
    
    def callback(indata, frames, time, status):
        nonlocal q
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    with sf.SoundFile(audFile, mode='x', samplerate=fs, channels=1) as file:
        with sd.InputStream(samplerate=fs, channels=1, callback=callback):
            with key.Events() as events:
                for event in events:
                    file.write(q.get())
                    if event.key == key.Key.esc:
                        break
    return audFile

def CapAudio() :
    audio = pyaudio.PyAudio()

    chosen_device_index = -1
    for x in range(0,audio.get_device_count()):
       info =audio.get_device_info_by_index(x)
       if info["name"] == "pulse":
           chosen_device_index = info["index"]

    stream = audio.open(
        format = pyaudio.paInt16, 
        channels=1, 
        rate=44100, 
        input = True, 
        frames_per_buffer = 1024, 
        input_device_index = chosen_device_index
    )
    keep_going = True
    frames = []
    try :
        th.Thread(target=key_capture_thread, args=(), name = 'key_capture_thread', daemon=True).start()
        while keep_going:
            data = stream.read(1024)
            frames.append(data)

    except KeyboardInterrupt:
        pass

    stream.stop_stream()
    stream.close()
    audio.terminate()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    audFile = f"audios/audiofile_{timestamp}.wav"
    write(audFile, 44100, np.array(frames))
    return audFile

if __name__ == "__main__":
    audFile = sound_device()
    text = whisper(audFile)
    print(text)