import sounddevice as sd
import numpy as np
import librosa

# Record 10 seconds of audio
def record_audio(duration=5, sample_rate=16000):
    print("Recording started...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    audio_data = np.squeeze(audio_data)
    return audio_data, sample_rate

# Extract pitch, intensity, and pace (speech rate)
def extract_features(audio, sr):
    # Pitch extraction using librosa
    pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr)
    pitch_values = np.max(pitches, axis=0)
    pitch_values = pitch_values[pitch_values > 0]  # Only non-zero pitches
    
    # Loudness (RMS) calculation
    rms = librosa.feature.rms(y=audio)
    avg_loudness = np.mean(rms)
    
    # Speech rate estimation (based on zero crossing rate)
    zero_crossings = librosa.feature.zero_crossing_rate(audio)
    avg_pace = np.mean(zero_crossings)
    
    return np.mean(pitch_values), avg_loudness, avg_pace

# Provide feedback based on pitch, loudness, and pace
def generate_feedback(pitch, loudness, pace):
    feedback = ""
    
    # Pitch feedback
    if pitch > 200:  # Higher pitch indicates excitement or stress
        feedback += "You sounded excited or stressed during the interview.\n"
    elif pitch < 100:
        feedback += "You seemed calm and relaxed.\n"
    
    # Loudness feedback
    if loudness > 0.05:  # Higher loudness can indicate confidence
        feedback += "You sounded confident throughout the interview.\n"
    else:
        feedback += "You may have been too quiet, indicating hesitation.\n"
    
    # Pace feedback
    if pace > 0.1:  # High pace can indicate nervousness or urgency
        feedback += "You were speaking quickly, which may indicate nervousness.\n"
    else:
        feedback += "Your speech pace was steady, indicating composure.\n"
    
    return feedback

# Main function to execute all steps
def main():
    # Step 1: Record audio
    audio, sr = record_audio()
    
    # Step 2: Extract features
    pitch, loudness, pace = extract_features(audio, sr)
    print(f"Pitch: {pitch:.2f} Hz, Loudness: {loudness:.4f}, Pace: {pace:.4f}")
    
    # Step 3: Generate feedback
    feedback = generate_feedback(pitch, loudness, pace)
    print("\nFeedback:\n", feedback)

if __name__ == "__main__":
    main()
