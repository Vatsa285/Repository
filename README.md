# Voice Emotion Detector

An AI project that fine-tunes a Wav2Vec2 speech-emotion model and serves it via a FastAPI backend with a React (Vite) frontend for recording audio and detecting emotions in speech.

## Key features
- Fine-tuning pipeline for Wav2Vec2 (audio emotion classification).
- FastAPI backend that accepts an uploaded audio file and returns predicted emotion + probabilities.
- React (Vite) frontend that records audio in-browser and calls the inference API.
- Example utilities to download / save Hugging Face pretrained models.

## Project layout
```
Dependencies                      # plain-text list of Python & runtime dependencies + link to model
README.md                         # (this file)
train.py                          # training script (data loading, training config, saving model)
load.py                           # helper: download Hugging Face model and save locally
pth.py                            # helper: convert/save model state_dict (.pth)
backend/
  main.py                         # FastAPI app; loads model and feature extractor and provides /predict/
my-emotion-detector/              # React (Vite) frontend
  package.json
  public/index.html
  src/
    App.js                         # UI: record audio, send to backend, show emotion
    index.js, index.css
```

## Dependencies / Requirements
The repo contains a `Dependencies` file listing the packages used. Core packages include:

Python (training + backend)
- transformers (Hugging Face)
- torch, torchaudio
- librosa, soundfile, pydub
- numpy, pandas
- scikit-learn
- datasets, evaluate
- fastapi, uvicorn, pydantic, starlette

Frontend
- node (>=16), npm/yarn
- react, react-dom, react-mic, vite

System
- ffmpeg (required by pydub) — install and add to PATH.

Note: There is no requirements.txt; see the Dependencies file and the commands below to install the needed packages.

## Quickstart — Backend (inference)
1. Prepare a Python environment:
```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
# .\venv\Scripts\Activate.ps1
```

2. Install core Python packages (example):
```bash
pip install transformers torch torchaudio librosa numpy pandas scikit-learn datasets evaluate fastapi uvicorn soundfile pydub
```
(Adjust torch install command for your CUDA/CPU setup as recommended on https://pytorch.org.)

3. Provide a saved fine-tuned model in Hugging Face save_pretrained format (model + feature extractor). Backend expects a folder matching the `model_path` in `backend/main.py`:
- Default in repository: `emotion_model_finetuned_v3`
- Options:
  - Use the provided Google Drive model (link in `Dependencies`) — download and place the folder at the path backend/main.py expects.
  - Or run your training (see Training section) to produce a folder, or use `load.py` to download a pretrained HF model and save it locally (it saves to `./my_emotion_model` by default).

4. Run the API:
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

5. Health / test endpoint:
- The inference endpoint is POST `/predict/` and accepts a multipart/form-data upload with a `file` field containing the audio file.

Example curl:
```bash
curl -X POST "http://127.0.0.1:8000/predict/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/audio.wav"
```

Response example (JSON):
```json
{
  "emotion": "happy",
  "probabilities": {
    "neutral": 0.01,
    "calm": 0.02,
    "happy": 0.80,
    "sad": 0.05,
    "angry": 0.03,
    "fearful": 0.02,
    "disgusted": 0.03,
    "surprised": 0.04
  },
  "status": "success"
}
```

Notes:
- The backend converts uploaded audio to WAV via pydub and loads with librosa at sr=16000 (resampling to 16kHz).
- If your model is saved in a different directory, update `model_path` in `backend/main.py` accordingly.

## Quickstart — Frontend (my-emotion-detector)
1. Install dependencies and run the dev server:
```bash
cd my-emotion-detector
npm install
npm run dev    # uses Vite (recommended)
# or: npm start  (package.json includes a "start": "react-scripts start", but the "dev" script uses vite)
```

2. The app UI records audio (via react-mic) and sends it to the API URL configured in `src/App.js`:
- Default: `http://127.0.0.1:8000/predict/`
- If backend runs on a different host/port, update the URL in `src/App.js`.

## Training (high-level)
- `train.py` loads the RAVDESS dataset path (look for dataset path inside `train.py`), maps emotion labels, prepares Hugging Face `datasets` splits, configures a Hugging Face `Trainer` for Wav2Vec2ForSequenceClassification and trains the model.
- Training outputs are saved to a directory (examples in the repo: `./emotion_model_finetuned_v3`, `./emotion_model_finetuned_v4` — check the `training_args` in `train.py` to confirm the exact `output_dir`).
- After training, ensure the saved model directory contains:
  - model files (pytorch_model.bin or similar, config.json)
  - feature extractor files (preprocessor_config.json or feature_extractor files)
  These are necessary for `Wav2Vec2ForSequenceClassification.from_pretrained()` and `Wav2Vec2FeatureExtractor.from_pretrained()` used by the backend.

Utilities:
- `load.py` downloads a Hugging Face pretrained model (`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`) and saves it to `./my_emotion_model`.
- `pth.py` demonstrates saving a model's state_dict to a `.pth` file; note `backend/main.py` expects the Hugging Face `save_pretrained` layout (not a single state_dict .pth). If you prefer .pth, you must modify `backend/main.py` to load state_dicts appropriately.

## Model storage / compatibility notes
- Backend uses `Wav2Vec2ForSequenceClassification.from_pretrained(model_path)` and `Wav2Vec2FeatureExtractor.from_pretrained(model_path)`. Provide a folder with Hugging Face `save_pretrained` output.
- If you have only a `.pth` state_dict, either convert it into HF save_pretrained format or update backend to load the architecture and load the state dict.

## Common issues & troubleshooting
- Audio conversion errors: Ensure ffmpeg is installed and on PATH (pydub uses ffmpeg for conversions).
- Model loading errors: Confirm model folder contains the HF files (config.json, pytorch_model.bin or similar) and feature extractor files.
- Mismatched sampling rate: Backend uses sr=16000 with librosa. If your audio uses another sampling rate, librosa will resample; very short audio can cause unstable behavior—use >1 second recommended.
- Frontend CORS: backend/main.py allows origins ["*"] by default; keep that in mind for production.

## Where to get the model
- The `Dependencies` file includes a Google Drive link to a model trained by the repository authors. Download and place the model folder at the path configured in `backend/main.py` (default `emotion_model_finetuned_v3`) or update `model_path` accordingly.

## Suggested improvements
- Add a `requirements.txt` (or `pyproject.toml`) for reproducible Python installs.
- Make `model_path` configurable via an environment variable in `backend/main.py`.
- Add Dockerfile(s) for backend and frontend for easier deployment.
- Add simple tests for API and model loading.

## Contributing
- Fork the repo, make changes, and open a PR.
- When changing model directory names or training outputs, update backend/main.py and README accordingly.

## License
No license file detected in the repository. If you want permissive reuse, add a LICENSE file (e.g., MIT) to the repo.

---

If you want, I can:
- Create a ready-to-commit README.md in this repository with this content.
- Add a requirements.txt generated from the `Dependencies` file.
- Make backend.model_path configurable via environment variables and open a PR with that change.
