import torch
from transformers import Wav2Vec2ForSequenceClassification

# Load pre-trained model
model = Wav2Vec2ForSequenceClassification.from_pretrained("./emotion_model_finetuned_v")

# Save the model
torch.save(model.state_dict(), "wav2vec2_emotion.pth")

print("Model saved as wav2vec2_emotion.pth")
