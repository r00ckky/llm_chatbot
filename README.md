# Multifunctional Python ChatBot

This Chatbot encompasses several scripts designed for various functionalities. Below is an overview of each component.

## Language Model Interaction (`Chatbot.py`)

This script defines two classes, `Llama` and `GPT`, for interacting with language models using the Hugging Face Transformers library and the OpenAI GPT-3.5-turbo API, respectively.

<br>

## Face Recognition and Information Script (`face.py`)

This script utilizes the `face_recognition` library and `OpenCV` for face recognition and information extraction. The script provides a `Face` class with methods for initializing face recognition, obtaining face information from an image, and saving new face information.

## Image Memory Management(`memory.py`)

This script is designed for managing image-related information, including face recognition, summarization, and data storage. It utilizes the `GPT` class from the `llm` module and the `Face` class for face recognition.

## Automatic Speech Recognition with Whisper(`stt.py`)

This script performs automatic speech recognition (ASR) using the OpenAI Whisper ASR model. It records audio using the `sounddevice` library and saves the recorded audio as a WAV file. The script uses the `transformers` library to access the Whisper ASR model and `pyautogui` for triggering the audio recording.

## Text To Speech(`tts.py`)
