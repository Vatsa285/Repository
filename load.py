from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

# Model name from Hugging Face
model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

# Download and save the model properly
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# Save locally
save_path = "./my_emotion_model"
feature_extractor.save_pretrained(save_path)
model.save_pretrained(save_path)

print("Model and feature extractor downloaded and saved successfully!")

