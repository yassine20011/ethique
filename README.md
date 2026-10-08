# Traffic Sign Detection and Classification (GTSRB)

Real-time traffic sign detection and classification for Advanced Driver Assistance Systems (ADAS) and autonomous vehicles using **YOLOv8** and a custom **PyTorch CNN** on the German Traffic Sign Recognition Benchmark (GTSRB).

---

## Performance Overview

| Model | Task | Metric | Score | Inference Speed |
| :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n** | Detection + Classification | **mAP@50** | **99.3%** | ~8.2 ms / image |
| **YOLOv8n** | Detection + Classification | **mAP@50-95** | **95.0%** | ~8.2 ms / image |
| **YOLOv8n** | Detection + Classification | **Precision** | **98.8%** | ~8.2 ms / image |
| **YOLOv8n** | Detection + Classification | **Recall** | **99.2%** | ~8.2 ms / image |
| **Custom CNN (PyTorch)** | Classification | **Test Accuracy** | **90.93%** | - |

---

## Dataset

This project utilizes the **German Traffic Sign Recognition Benchmark (GTSRB)**:
- **Total Images:** 39,200+ images
- **Classes:** 43 distinct traffic sign classes (speed limits, danger warnings, prohibitions, priority signs, etc.)
- **Split:**
  - Training: 31,248 images
  - Validation: 7,960 images
- **Preprocessing:** Conversion from original `.ppm` format to `.jpg`, ROI coordinates mapped to normalized YOLO bounding boxes `(x_center, y_center, width, height)`.

---

## Architecture & Approaches

### 1. YOLOv8 Nano (Detection & Classification) - `ethique_v2.ipynb`
- **Backbone:** Ultralytics YOLOv8n pre-trained weights.
- **Input Resolution:** 416x416.
- **Training Config:** Batch size of 32, 15 epochs, SGD/AdamW optimizer with auto-anchor matching.
- **Key Advantage:** Simultaneously localizes and identifies traffic signs in under 10 ms, meeting real-time automotive ADAS latency requirements.

### 2. Custom PyTorch CNN (Classification Benchmark) - `ethique.ipynb`
- **Structure:** Multi-layer Convolutional Neural Network with Batch Normalization, Max Pooling, and Dropout for regularization.
- **Training Config:** CrossEntropyLoss, Adam optimizer, 10 epochs.
- **Performance:** 90.93% accuracy on the GTSRB test split.

---

## Project Structure

```
├── ethique_v2.ipynb    # YOLOv8 pipeline: data formatting, training, validation & inference
├── ethique.ipynb       # Custom PyTorch CNN baseline classification pipeline
├── pyproject.toml      # Project dependencies managed via uv
└── README.md           # Project documentation and benchmark results
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (recommended for training)

### Installation

Using `uv`:
```bash
uv sync
```

Or using standard `pip`:
```bash
pip install ultralytics torch torchvision torchaudio pandas matplotlib opencv-python pillow tqdm
```

### Running the Notebooks
Launch Jupyter or Google Colab and open:
- `ethique_v2.ipynb` for the full YOLOv8 detection pipeline.
- `ethique.ipynb` for the PyTorch CNN baseline.

---

## License

This project is licensed under the MIT License.
