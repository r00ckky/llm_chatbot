import whisper

def TranscribeAud(Aud) :

    model = whisper.load_model("base")
    result = model.transcribe(Aud, fp16 = False)

    return result["text"]