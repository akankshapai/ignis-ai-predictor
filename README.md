# 🛰️ IGNIS-AI: Satellite Wildfire Predictor

An interactive, deep learning-powered computer vision application that classifies satellite imagery to assess regional wildfire threats in real time. 

---

## 🚀 Live Demo

Experience the active deployment and upload your own orbital imagery feeds here:
👉 **[Launch IGNIS-AI Dashboard](https://huggingface.co/spaces/akankshapai/ignis-ai-predictor)**

---

## 🛠️ Features & Architecture

* **Real-Time Classification:** Upload any satellite landscape imagery (`PNG`/`JPG`) to evaluate environmental conditions instantly.
* **Sensitivity Threshold Adjustments:** Dynamic slider controls allowing operators to manually shift detection sensitivity markers between conservative and aggressive parameters.
* **Production-Optimized Inference:** Employs an ultra-lightweight, quantized `.tflite` model format ensuring high computational speeds under tight memory limitations.
* **Containerized Deployment:** Packaged meticulously via a custom Python 3.10 Linux Docker runtime to guarantee architecture stability on the cloud.

---

## 🧱 Project Directory Structure

```text
├── src/
│   ├── app.py                   # Main interactive Streamlit application logic
│   └── wildfire_model.tflite    # Quantized deep learning model weight file
├── Dockerfile                   # Custom Python 3.10 container configuration 
├── requirements.txt             # Lightweight package dependency matrix
└── README.md                    # Project presentation & documentation
