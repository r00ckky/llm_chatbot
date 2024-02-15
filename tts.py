from transformers import AutoProcessor, AutoModel
import scipy

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = AutoModel.from_pretrained("suno/bark-small")

inputs = processor(
    text=["Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe."],
    return_tensors="pt",
)

speech_values = model.generate(**inputs, do_sample=True)

sampling_rate = model.config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=speech_values, data=speech_values.cpu().numpy().squeeze())


