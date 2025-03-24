from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import librosa
import io
from pydub import AudioSegment
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load fine-tuned model and feature extractor
model_path = "emotion_model_finetuned_v3"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

emotion_labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]

@app.post("/predict/")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        # Read file into BytesIO
        contents = await file.read()
        audio_buffer = io.BytesIO(contents)

        # Convert to WAV if necessary
        try:
            audio_segment = AudioSegment.from_file(audio_buffer)
            audio_buffer = io.BytesIO()
            audio_segment.export(audio_buffer, format="wav")
            audio_buffer.seek(0)
        except Exception as e:
            return {"status": "error", "message": f"Audio conversion failed: {str(e)}"}

        # Reload the audio buffer for librosa
        audio_buffer.seek(0)
        try:
            y, sr = librosa.load(audio_buffer, sr=16000)
        except Exception as e:
            return {"status": "error", "message": f"Librosa loading failed: {str(e)}"}

        # Extract features
        inputs = feature_extractor(y, sampling_rate=sr, return_tensors="pt")

        # Make prediction
        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_label = torch.argmax(probabilities, dim=-1).item()

        predicted_emotion = emotion_labels[predicted_label]
        emotion_probabilities = {emotion_labels[i]: float(probabilities[0][i]) for i in range(len(emotion_labels))}

        print(f"API Response: {predicted_emotion}, Probabilities: {emotion_probabilities}")

        return {"emotion": predicted_emotion, "probabilities": emotion_probabilities, "status": "success"}

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e), "status": "failure"}
