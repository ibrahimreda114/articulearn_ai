import os
import requests
import sounddevice as sd
from scipy.io.wavfile import write
from requests.exceptions import RequestException

# ==========================================
# Configuration Constants
# ==========================================
RECORDING_DURATION = 3  # Changed to 3 seconds
SAMPLE_RATE = 16000     
SAVED_AUDIO_FILE = "my_recorded_practice.wav" # The file will be saved with this name
API_ENDPOINT = "http://127.0.0.1:8000/evaluate"

def record_audio(filename: str, duration: int, sample_rate: int) -> None:
    """
    Records audio from the default microphone and saves it as a WAV file.
    """
    print(f"\n🎤 Recording... Please speak now (Duration: {duration} seconds)")
    try:
        recording = sd.rec(
            int(duration * sample_rate), 
            samplerate=sample_rate, 
            channels=1, 
            dtype='int16'
        )
        sd.wait()  
        write(filename, sample_rate, recording)
        print(f"✅ Recording saved locally as: {filename}")
    except Exception as e:
        print(f"❌ Error during recording: {e}")
        raise

def evaluate_pronunciation(target_word: str, filename: str) -> None:
    """
    Sends the recorded audio and target word to the API for evaluation.
    """
    print("⏳ Analyzing your pronunciation...")
    
    try:
        with open(filename, "rb") as audio_file:
            files = {"file": (filename, audio_file, "audio/wav")}
            data = {"target_word": target_word}
            
            response = requests.post(API_ENDPOINT, files=files, data=data)
            response.raise_for_status() 
            
            result = response.json()
            
            print("\n" + "="*40)
            print("🎯 EVALUATION RESULTS")
            print("="*40)
            print(f"Target Word : {result.get('target')}")
            print(f"Predicted   : {result.get('predicted')}")
            
            scores = result.get('scores', {})
            print(f"Final Score : {scores.get('final')}%")
            print(f"Feedback    : {result.get('feedback')}")
            
            mistakes = result.get('mistakes', [])
            if mistakes:
                print(f"Mistakes    : {', '.join(mistakes)}")
            print("="*40 + "\n")
            
    except RequestException as e:
        print(f"❌ API connection failed: {e}")

def main():
    try:
        target_word = input("📝 Enter the word to practice: ").strip()
        
        if not target_word:
            print("⚠️ No word entered.")
            return

        # 1. Record the audio (will be saved in the same folder)
        record_audio(SAVED_AUDIO_FILE, RECORDING_DURATION, SAMPLE_RATE)
        
        # 2. Send to API and print results
        evaluate_pronunciation(target_word, SAVED_AUDIO_FILE)
        
        print(f"ℹ️  You can find the audio file '{SAVED_AUDIO_FILE}' in your folder to listen to it.")

    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()