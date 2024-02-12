import torch
from TTS.api import TTS
from scipy.io.wavfile import write
import numpy as np

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
wav = tts.tts(text="This video is sponsored by Nord VPN! And you all can get! Free! 3 months subcription using my code.", speaker_wav="gauri_sample.wav", language="en")
write("output_gauri.wav", 22050,np.array(wav))