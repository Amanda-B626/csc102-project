import wave
import librosa
import numpy as np
import sounddevice as sd
from sklearn.metrics.pairwise import cosine_similarity

def read_wav_file(file_path):
    try:
        with wave.open(file_path,'rb') as wav_file:
            num_channels = wav_file.getnchannels()
            # Get the sample width in bytes
            sample_width = wav_file.getsampwidth()
            # Get the frame rate
            frame_rate = wav_file.getframerate()
            # Get the number of frames
            num_frames = wav_file.getnframes()
            # Read all frames as bytes
            frames = wav_file.readframes(num_frames)

            print(f"Number of Channels: {num_channels}")
            print(f"Sample Width: {sample_width} bytes")
            print(f"Frame Rate: {frame_rate} Hz")
            print(f"Number of Frames: {num_frames}")
            print(f"Frames (as bytes): {frames[:100]}...")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except wave.Error as e:
        print(f"Error reading WAV file: {e}")

wake_word_audio, sr = librosa.load("fuck.wav")
wake_word_mfccs = librosa.feature.mfcc(y=wake_word_audio, sr=sr, n_mfcc=13)
def extract_mfccs(audio, sr):
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    mfccs = extract_mfccs(indata[:, 0], sr)
    similarity = cosine_similarity(wake_word_mfccs.T, mfccs.T).max()
    if similarity > 0.7:
      print("Wake word detected")
      
with sd.InputStream(callback=audio_callback, channels=1, samplerate=sr):
  print("Listening for wake word...")
  sd.sleep(1000000) # Keep the stream active
  
  
# Example usage:
file_path = 'fuck.wav'  # Replace with your file path
read_wav_file(file_path)