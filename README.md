# Deepfake Face and Voice Detection

A full-stack deepfake detection project that analyzes both video (facial manipulation) and audio (voice spoofing) using trained TensorFlow/Keras models, a FastAPI backend, and a Streamlit user interface. The system is designed to provide a practical, end-to-end workflow for uploading media files and receiving a probability-based verdict of whether the content is likely real or fake.

## Project Overview

This project combines two separate detection pipelines:

- Face detection for videos using sampled frames, face cropping, and a CNN-based classification model.
- Voice detection for audio clips using MFCC feature extraction and a trained deepfake voice model.

The repository includes:

- Training notebooks for the face and voice models.
- A FastAPI backend for inference.
- A Streamlit frontend for upload and analysis.
- Pretrained model files in the `models/` directory.
- Docker support for containerized deployment.

## Why This Project Matters

Synthetic media detection is increasingly important for verifying the authenticity of videos and audio files in real-world scenarios such as misinformation detection, media forensics, and content moderation. This project demonstrates one approach to building a detector that checks both visual and audio cues.

## System Architecture

The project follows a simple but effective structure:

- `notebooks/` contains the experimental model-building workflows.
- `backend/` hosts the API logic and inference code.
- `frontend/` contains the user-facing Streamlit dashboard.
- `models/` stores the trained model artifacts.
- `Dockerfile` and `compose.yaml` allow easy deployment.

## Repository Structure

```text
Deepfake-Face-And-Voice-Detection/
├── backend/
│   ├── __init__.py
│   ├── inference.py
│   └── main.py
├── frontend/
│   └── app.py
├── models/
│   ├── deepfake_face_detection.keras
│   └── deepfake_voice_model.keras
├── notebooks/
│   ├── deepfake-face_detection.ipynb
│   └── deep-fake-voice-recognition-using-cnn.ipynb
├── .gitignore
├── Dockerfile
├── README.md
├── compose.yaml
├── requirements.txt
└── LICENSE (if present in your repo)
```

## Step-by-Step Workflow

### 1. Notebook-based model development

The project includes two notebooks:

- `notebooks/deepfake-face_detection.ipynb`
- `notebooks/deep-fake-voice-recognition-using-cnn.ipynb`

These notebooks implement the exploratory work used to train and validate the CNN-based detection models. The backend logic mirrors the same preprocessing logic used during training, ensuring that the deployed model behaves consistently with the original experiments.

### 2. Video processing pipeline

For video inference:

1. The uploaded file is validated for supported formats (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`).
2. A fixed number of frames are sampled from the video.
3. OpenCV detects the main face region in each frame.
4. The detected face is cropped and resized to `224x224`.
5. Frames are converted to RGB and normalized.
6. The sequence is passed through the face model.
7. The output is converted into a probability score, label, and confidence score.

The code uses a sequence length of 20 frames and a threshold of 0.5 for classification. If the predicted fake probability is greater than or equal to 0.5, the model outputs `FAKE`.

### 3. Audio processing pipeline

For audio inference:

1. The uploaded file is validated for supported formats (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`).
2. The audio is loaded at a sampling rate of 16,000 Hz.
3. MFCC features are extracted using Librosa.
4. The MFCC matrix is padded/truncated to a fixed length.
5. The normalized spectrogram-like representation is fed into the voice classifier.
6. The system outputs a probability score and a real/fake decision.

The audio model analyzes up to around 16 seconds of audio by design, which is useful for rapid inference and consistent model input sizing.

### 4. Backend API with FastAPI

The backend is implemented in `backend/main.py` and `backend/inference.py`.

Available API endpoints:

- `GET /health` — checks whether models are loaded and reports backend status.
- `POST /predict/video` — accepts a video upload and returns fake probability, label, and confidence.
- `POST /predict/audio` — accepts an audio upload and returns fake probability, label, and confidence.

The server loads both trained models at startup, while keeping the service alive even if one model cannot be loaded.

### 5. Streamlit frontend

The `frontend/app.py` file provides a simple dashboard with:

- A sidebar for API configuration and health status.
- A video detection tab.
- An audio detection tab.
- Result cards with probability, confidence, and threshold information.

This makes it easy for non-technical users to test the model by uploading a file directly in the browser.

### 6. Deployment using Docker and Compose

The project includes:

- `Dockerfile` for building the backend image.
- `compose.yaml` for running both backend and frontend services.

This is useful for reproducible deployment and local testing in a consistent environment.

## Model Information

The repository includes the following trained model artifacts:

- `models/deepfake_face_detection.keras`
- `models/deepfake_voice_model.keras`

These models are designed for inference in the project’s FastAPI backend and are loaded at runtime from the `models/` directory.

## Key Findings and Observations

From the implementation and project design, several important findings stand out:

1. The solution is multi-modal.
   - It does not rely on a single signal; it combines visual and audio cues to improve detection coverage.

2. Video detection is frame-based.
   - The face model samples and analyzes a limited number of frames rather than processing the full video in one pass, which makes inference more efficient.

3. Face detection is assisted by OpenCV.
   - The system detects a face first and crops the face region before classification, improving relevance for deepfake detection.

4. Audio classification relies on MFCC features.
   - MFCC extraction is a common and effective approach for audio-based spoofing detection because it captures characteristic spectral information.

5. Results are probability-driven.
   - The model returns a fake probability and confidence score rather than a deterministic claim, which is appropriate for AI-assisted forensic decisions.

6. Threshold-based decision making is applied.
   - The system uses a threshold of `0.5` to classify content as `FAKE` or `REAL`.

7. Reliability checks are included.
   - In the face pipeline, if too few frames contain a detectable face, the UI warns that the result may be less reliable.

8. The project is structured for real-world use.
   - It includes both a human interface and an API, which means it can be used for direct research, demos, and integration into other systems.

## Setup and Installation

### Prerequisites

- Python 3.11+
- pip
- Optional: Docker and Docker Compose

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Run the frontend

```bash
streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501
```

Then open:

- Backend API: `http://localhost:8000`
- Frontend UI: `http://localhost:8501`

## Docker Deployment

Build and run the stack with Docker Compose:

```bash
docker compose up --build
```

This starts:

- Backend on port `8000`
- Frontend on port `8501`

## Usage Example

### Video detection

Upload a video to the Streamlit app or send a request to the API:

```bash
curl -X POST "http://localhost:8000/predict/video" \
  -F "file=@sample_video.mp4"
```

### Audio detection

```bash
curl -X POST "http://localhost:8000/predict/audio" \
  -F "file=@sample_audio.wav"
```

## Limitations

This project is a practical proof-of-concept and should be treated as an experimental detector, not as a definitive forensic tool. Important limitations include:

- Model performance depends on the quality and diversity of the dataset used during training.
- Results are probabilistic and should be interpreted with caution.
- Face detection may fail in poor-quality or heavily occluded media.
- Audio detection can be affected by compression, background noise, and recording artifacts.
- The final decision should be used as an AI-assisted signal, not as sole evidence.

## Future Improvements

Possible enhancements include:

- Expanding the dataset to include more diverse real and fake examples.
- Adding robust evaluation metrics such as precision, recall, F1-score, and ROC-AUC.
- Improving the UI with confidence visualizations and historical results.
- Supporting batch inference for multiple files.
- Integrating explainability tools to show which cues contributed to a fake prediction.

## Conclusion

This repository presents a full deepfake detection pipeline for both video and audio, from model development to deployment. It demonstrates how trained detection models can be wrapped in a web interface and API so that users can upload media files and receive actionable predictions in real time.

The project reflects a strong end-to-end workflow: model experimentation, preprocessing alignment, deployment, and user-facing inference. It is a solid starting point for further research, model refinement, and production-oriented deployment.

## License

Please add your preferred license here if needed.

## Contact

If you want to extend or improve this project, open an issue or contribute a pull request to the repository.
