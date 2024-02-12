import sounddevice as sd
import soundfile as sf
import time
import numpy as np
from transformers import pipeline
import pyautogui as pag
import queue
import sys
import tempfile
assert np

whisper = pipeline(
    'automatic-speech-recognition',
    model = 'openai/whisper-tiny',
    device = 0
)

def record_audio():
    q = queue.Queue()
    fs = 44100
    timestamp = time.strftime("%Y%m%d")
    audFile = tempfile.mktemp(prefix=f"audiofile_{timestamp}", suffix='.wav', dir="audios")

    def callback(indata, frames, time, status):
        nonlocal q
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    try:
        with sf.SoundFile(audFile, mode = 'x', samplerate = fs, channels = 1) as file:
            with sd.InputStream(samplerate = fs, channels = 1, callback = callback):
                pag.press('space')
                while not pag.keyUp('space'):
                    file.write(q.get())
                pag.press('space')
    finally:
        return audFile
           
if __name__ == "__main__":
    while True:
        record_audio()
        