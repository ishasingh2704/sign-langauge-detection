#  Sign Language detector


A premium, end-to-end multimodal translation application that leverages real-time computer vision, local Large Language Models (LLMs), and adaptive Text-to-Speech (TTS) to bridge the communication gap for American Sign Language (ASL) speakers. 

The system tracks hand gestures to recognize spelling, processes facial expressions to identify user emotion, refines character sequences into coherent speech using a local LLM, translates the refined text into 40+ languages, and speaks the translated output with voice inflections modulated by the user's emotion.

---

## 🌟 Key Features

* **Real-Time ASL Alphabet Recognition**: High-fidelity detection of ASL fingerspelling (letters A-Z) using MediaPipe Hand tracking and a custom, lightweight classification model.
* **Synchronous Emotion Analysis**: DeepFace-based facial expression analysis running asynchronously in a separate worker thread to track user emotion without bottlenecking camera frames.
* **AI-Powered Text Refinement**: Integrates a local LLM to translate raw letter buffers (e.g., `"h-e-l-l-o-o-w-o-r-l-d"`) into proper, punctuated English sentences.
* **Adaptive Emotional Speech (TTS)**: Leverages Piper TTS to generate synthesized audio streams, automatically adjusting vocal speed, pitch, and voice noise to match the user's detected emotion.
* **Multilingual Translation**: Instant translation of refined sentences into over 40 languages (Spanish, French, German, Japanese, Arabic, etc.) using the `deep-translator` library.
* **MongoDB Session Logging**: Automatically archives complete session translation history, including raw character buffers, LLM outputs, emotional logs, and translation parameters.

---

## 🛠️ Architecture & Pipeline

### 1. Gesture Classification Pipeline
* **Landmark Extraction**: Detects 21 2D landmarks (x, y) per hand.
* **Preprocessing**: Subtracts the wrist coordinate to ensure spatial translation invariance, flattens coordinates into a 42-element vector, and divides by the maximum absolute coordinate value in the vector to achieve scale invariance.
* **Classification Model**: A deep neural network utilizing `Mish` activation layers, Batch Normalization, Dropout regularization, and a Softmax output layer, converted to **TensorFlow Lite (TFLite)** for ultra-low latency CPU execution.

### 2. Facial Expression Alignment & Inference
* **Face Localization**: Isolates face bounding boxes using MediaPipe Face Detection.
* **Alignment**: Rotates the cropped face frame based on eye keypoint angles to resolve tilt, resizes it to 48x48 pixels, and runs grayscale histogram equalization to resolve poor lighting conditions.
* **Multi-Sample Smoothing**: Employs rolling-buffer weighted voting with recency bias to deliver a stable, noise-free emotion assessment.

### 3. Speech Inflection Parameters
The Piper TTS synthesis scales voice configuration in real-time according to emotion profiles:
* **Sad**: Slowed rate (`0.60x` speed, `0.40` noise scale).
* **Angry**: Accelerated rate (`1.20x` speed, `1.20` noise scale).
* **Happy**: Upbeat rate (`1.05x` speed, `1.50` noise scale).
* **Neutral**: Standard rate (`1.00x` speed, `0.66` noise scale).

---

## 🚀 Setup & Installation

### Prerequisites
* Python 3.9 - 3.11
* MongoDB installed and running locally on port `27017`
* LM Studio (or a compatible local OpenAI endpoint)

### 1. Clone & Prepare Environment
```bash
# Initialize and install dependency packages inside a virtual environment
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set Up LLM Endpoint (LM Studio)
1. Open **LM Studio**.
2. Search and download the `qwen2.5-7b-instruct-1m` model (or any model of your choice).
3. Start the local server endpoint on port `1234` (Base URL: `http://localhost:1234/v1`).

### 3. Download TTS Voices
Ensure ONNX voice files and matching configuration JSON files are downloaded and placed inside the `voices/` directory. For example:
* `voices/en_US-amy-medium.onnx`
* `voices/en_US-amy-medium.onnx.json`
* `voices/en_US-sam-medium.onnx`
* `voices/en_US-sam-medium.onnx.json`

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
MONGODB_URI=mongodb://127.0.0.1:27017
MONGODB_DATABASE=asl_detection
MONGODB_COLLECTION=letter_sequences
MONGODB_REFINED_COLLECTION=refined_sentences
BUFFER_MAX_SIZE=500
```

---

## 💻 Running the App

1. Ensure MongoDB and your LM Studio local server are running.
2. Launch the Streamlit application:
   ```bash
   streamlit run streamlit_app.py
   ```
3. Use the sidebar controls to:
   * Select your webcam device.
   * Toggle **Translation** and select target languages.
   * Enable **Text-to-Speech** and choose your preferred ONNX voice model.
   * Start/stop the camera feed, and view real-time metrics, character buffers, and corrected translation sequences.

---

## 📊 Dataset & Model Details
* **ASL Dataset**: Located at `asl_classifier/dataset.csv`, consisting of 36,430 rows of normalized landmark data representing 26 classes (A-Z).
* **ASL Model Performance**: The keypoint classifier model achieves a **86.10%** test accuracy (evaluated on 9,108 validation samples) with a sparse categorical cross-entropy loss of **0.7396**.
