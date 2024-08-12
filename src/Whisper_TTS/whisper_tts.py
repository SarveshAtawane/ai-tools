import torch
from whisperspeech.pipeline import Pipeline
import torchaudio
import os
import requests

def download_reference_audio(url, output_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, "wb") as file:
            file.write(response.content)
        print(f"File downloaded and saved to {output_path}")
        return True
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
        return False

def initialize_pipeline():
    return Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-tiny-en+pl.model')

def text_to_speech(pipe, text, reference_audio_path=None, cps=10.5):
    if reference_audio_path:
        audio = pipe.generate(text, lang='en', cps=cps, speaker=reference_audio_path)
    else:
        audio = pipe.generate(text, lang='en', cps=cps)
    
    audio_cpu = audio.cpu().squeeze()
    if audio_cpu.dim() == 1:
        audio_cpu = audio_cpu.unsqueeze(0)
    
    return audio_cpu

def save_audio(audio, output_file, sample_rate=24000):
    torchaudio.save(output_file, audio, sample_rate=sample_rate, encoding="PCM_F")
    print(f"Generated audio file: {output_file}")

def transcribe(text, output_file, use_voice_cloning=False, reference_audio_path=None):
    if use_voice_cloning and not reference_audio_path:
        print("Error: Voice cloning requested but no reference audio path provided.")
        return False
    
    if use_voice_cloning and not os.path.exists(reference_audio_path):
        print(f"Error: Reference audio file not found at {reference_audio_path}")
        return False
    
    pipe = initialize_pipeline()
    
    audio = text_to_speech(pipe, text, reference_audio_path if use_voice_cloning else None)
    
    save_audio(audio, output_file)
    
    return True

if __name__ == "__main__":
    text_to_transcribe = "This is a test of the WhisperSpeech text-to-speech system with optional voice cloning."
    output_file = "output.wav"
    # you can add a add your audio download link from drive for reference audio
    reference_audio_url = "https://drive.google.com/uc?export=download&id=1P5kM5-U9tk3bdw309ybIbkQoxWw04YpM"
    reference_audio_path = "reference_audio.wav"

    # Download reference audio if it doesn't exist
    if not os.path.exists(reference_audio_path):
        download_reference_audio(reference_audio_url, reference_audio_path)

    # Without voice cloning
    transcribe(text_to_transcribe, "output_without_cloning.wav")
    
    # you can add your audio file path as reference_audio_path as wav file
    transcribe(text_to_transcribe, "output_with_cloning.wav", use_voice_cloning=True, reference_audio_path=reference_audio_path)