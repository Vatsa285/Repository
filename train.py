import os
import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (AutoFeatureExtractor, Wav2Vec2ForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
import evaluate
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
import torch.nn.functional as F

#  Load Pretrained Feature Extractor & Model
model_path = "./emotion_model_finetuned_v3"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)

#  Correct RAVDESS Emotion Labels Mapping
ravdess_emotion_mapping = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

#  Correct Label to Index Mapping (Matches Model)
label_mapping = {
    "neutral": 0, "calm": 1, "happy": 2, "sad": 3,
    "angry": 4, "fearful": 5, "disgust": 6, "surprised": 7
}

#  Load Audio Dataset
data_path = "C:\\Users\\raksh\\Desktop\\voice_detection\\dataset\\Audio_Speech_Actors_01-24"

data = []
for root, _, files in os.walk(data_path):
    for file in files:
        if file.endswith(".wav"):
            parts = file.split("-")  # Extract parts of filename
            emotion_code = parts[2]  # Third element is the emotion code
            label = ravdess_emotion_mapping.get(emotion_code, "unknown")
            if label in label_mapping:  # Ensure label is valid
                data.append({"path": os.path.join(root, file), "label": label})

df = pd.DataFrame(data)
ravdess_dataset = Dataset.from_pandas(df)

# Audio Preprocessing
def preprocess_audio(example):
    waveform, sample_rate = torchaudio.load(example["path"])

    # Convert Stereo to Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    if sample_rate != 16000:
        waveform = librosa.resample(waveform.numpy(), orig_sr=sample_rate, target_sr=16000)
        waveform = torch.tensor(waveform)

    # Normalize Waveform
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)

    # Extract Features
    features = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)

    # Store Extracted Features
    example["input_values"] = features.input_values[0].numpy()
    return example

ravdess_dataset = ravdess_dataset.map(preprocess_audio, remove_columns=["path"])

# Map Labels to Integers (Ensuring Correct Encoding)
def map_labels(example):
    example["label"] = label_mapping[example["label"]]
    return example

ravdess_dataset = ravdess_dataset.map(map_labels)

# Train-Test Split
ravdess_dataset = ravdess_dataset.train_test_split(test_size=0.2)
train_dataset, val_dataset = ravdess_dataset["train"], ravdess_dataset["test"]

# Compute Class Weights (Fix Dataset Imbalance)
labels = [ex["label"] for ex in train_dataset]
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to("cuda")

# Weighted Focal Loss (Fix Overfitting & Class Imbalance)
class WeightedFocalLoss(nn.Module):
    def __init__(self, weights, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = gamma
        self.weights = weights

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, weight=self.weights, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()
        return focal_loss

weighted_focal_loss = WeightedFocalLoss(class_weights)

# Define Loss Function for Trainer
def compute_loss(model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    loss = weighted_focal_loss(outputs.logits, labels)
    return (loss, outputs) if return_outputs else loss

# Evaluation Metrics
accuracy_metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Data Collator
data_collator = DataCollatorWithPadding(feature_extractor, return_tensors="pt")

# Optimized Training Arguments
training_args = TrainingArguments(
    output_dir="./emotion_model_finetuned_v4",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    push_to_hub=False,
    load_best_model_at_end=True,
    fp16=True
)

# Trainer with Improved Settings
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train Model
trainer.train()

# Save Fine-Tuned Model
model.config.id2label = {v: k for k, v in label_mapping.items()}
model.config.label2id = label_mapping
model.save_pretrained("./emotion_model_finetuned_v4")
feature_extractor.save_pretrained("./emotion_model_finetuned_v4")
print("Fine-tuning completed and model saved!")

# Evaluate Final Model
results = trainer.evaluate()
print(results)

# Verify Model Label Mapping
print("Model id2label:", model.config.id2label)
print("Model label2id:", model.config.label2id)
 
   
