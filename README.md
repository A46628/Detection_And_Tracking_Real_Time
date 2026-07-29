# Defense Drone System with Real-time Object Detection and Target Tracking

Autonomous aerial surveillance and target tracking powered by Deep Learning and System-on-Chip FPGA acceleration.

---

## Table of Contents
- [About The Project](#about-the-project)
- [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Tracking-by-Detection Pipeline](#tracking-by-detection-pipeline)
  - [Supported Trackers](#supported-trackers)
- [Results & Benchmarks](#results--benchmarks)
  - [1. Detection Model Validation (GPU)](#1-detection-model-validation-gpu)
  - [2. Detection Model Validation (DPU / Embedded FPGA)](#2-detection-model-validation-dpu--embedded-fpga)
  - [3. DPU Latency & Pipeline Breakdown](#3-dpu-latency--pipeline-breakdown)
  - [4. System Results (Detection + Tracking)](#4-system-results-detection--tracking)
- [Contributing](#contributing)
- [Contact and Acknowledgements](#contact-and-acknowledgements)

---

## About The Project

This project focuses on the development of an embedded and autonomous UAV defense system capable of performing real-time object detection and multi-target tracking under the **Tracking-by-Detection** paradigm.

To enable onboard decision-making and eliminate reliance on external communications, deep learning object detection models (such as **YOLOv7-tiny**) are optimized and quantized to 8-bit integer (**INT8**) precision using the **AMD Vitis AI** flow. The inference graph is deployed directly onto a Deep Learning Processing Unit (**DPU**) running on an **AMD Kria™ KV260 Vision AI Starter Kit** SoC FPGA.

### Key Features:
- **Onboard Edge AI:** Real-time INT8 model acceleration using AMD Xilinx DPU.
- **Custom Military Dataset:** Detection capability across 12 distinct classes (soldiers, military tanks, aircraft, weaponry, etc.).
- **Multiple Tracking Algorithms:** Integration with Kalman Filter and Hungarian algorithm-based trackers (SORT, DeepSORT, ByteTrack).
- **High Energy Efficiency:** Operates under low power consumption (~5.4W) suitable for drone platforms.

---

## Built With

This project integrates the following frameworks, platforms, and toolchains:

- **PyTorch** (Model training and validation)
- **YOLO Series** (YOLOv7, YOLOv8, YOLO11)
- **AMD Vitis AI 3.5** (Model quantization and DPU compilation)
- **AMD Kria™ KV260 Vision AI Starter Kit** (FPGA Target Hardware)
- **OpenCV & Python/C++** (Image processing and state tracking)

---

## Getting Started

### Prerequisites

#### Host PC / Training Server
- Linux OS (Ubuntu 20.04/22.04 recommended)
- NVIDIA GPU with CUDA support
- Anaconda / Miniconda
- Docker (required for Vitis-AI toolchain execution)

#### Target Hardware
- AMD Kria™ KV260 Vision AI Starter Kit
- Petalinux / Ubuntu Image for Kria KV260 with Vitis AI Runtime (VART) installed

---

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/seu-usuario/Defense-Drone-System.git
   cd Defense-Drone-System
   Set Up Python Environment:
   conda create -n drone-env python=3.8 -y

2. **Set Up Python Environment
   ```bash
    conda create -n drone-env python=3.8 -y
    conda activate drone-env
    pip install -r requirements.txt
   
3. **Train Model 
   ```bash
    python train.py --workers 8 --device 0 --batch-size 16 --data data/military_dataset.yaml --img 640 640 --cfg cfg/deploy/yolov7-tiny.yaml --weights '' --name yolov7-tiny-military

 ###Validate Model Metrics on Dataset

   To evaluate the model's accuracy (mAP, Precision, Recall) on the validation or test set:

    ```bash
  # Evaluate on Validation set
    python val.py --weights runs/train/yolov7-tiny-military/weights/best.pt --data data/military_dataset.yaml --img 640 --device 0
  
  # Evaluate on Test set with class-wise details
    python val.py --weights runs/train/yolov7-tiny-military/weights/best.pt --data data/military_dataset.yaml --img 640 --task test --verbose --device 0


### Detection Model Validation (GPU)

| Model | mAP 50 | mAP 50-95 | Precision | Recall | FPS | Speed (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv7-tiny** | 0.387 | 0.212 | 0.662 | 0.404 | **435** | **2.3** |
| **YOLOv7** | **0.629** | 0.424 | 0.683 | **0.610** | 278 | 3.6 |
| **YOLOv7x** | 0.517 | 0.323 | **0.716** | 0.494 | 167 | 6.0 |
| **YOLOv8** | 0.625 | 0.448 | 0.621 | 0.578 | 294 | 3.4 |
| **YOLO11n** | 0.560 | 0.375 | 0.695 | 0.502 | 370 | 2.7 |


## Usage
### Run Inference with Trained Weights (GPU / PC)

- **Single Image:**
  ```bash
  python detect.py --weights runs/train/yolov7-tiny-military/weights/best.pt --conf 0.25 --img-size 640 --source path/to/image.jpg

- **Video File:**
  ```bash
  python detect.py --weights runs/train/yolov7-tiny-military/weights/best.pt --conf 0.25 --img-size 640 --source path/to/video.mp4

 - **Webcam:**
  ```bash
   python detect.py --weights runs/train/yolov7-tiny-military/weights/best.pt --conf 0.25 --img-size 640 --source 0

