# Defense Drone System with Real-time Object Detection and Target Tracking

Autonomous aerial surveillance and target tracking powered by Deep Learning and System-on-Chip FPGA acceleration.

This project presents an embedded UAV perception system based on the **Tracking-by-Detection** paradigm. The system combines real-time object detection using YOLO models with multi-object tracking algorithms and hardware acceleration through an AMD Kria™ KV260 Vision AI Starter Kit.

The main objective is to achieve real-time detection and tracking on resource-constrained edge hardware while maintaining low latency, low power consumption, and sufficient detection and tracking accuracy.

---

## Table of Contents

* [About The Project](#about-the-project)
* [System Architecture](#system-architecture)
* [Dataset](#dataset)
* [Built With](#built-with)
* [Getting Started](#getting-started)

  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Training](#training)
* [Usage](#usage)

  * [GPU Inference](#gpu-inference)
  * [Tracking-by-Detection](#tracking-by-detection)
  * [Embedded Deployment](#embedded-deployment)
* [Tracking-by-Detection Pipeline](#tracking-by-detection-pipeline)

  * [Supported Trackers](#supported-trackers)
* [Results & Benchmarks](#results--benchmarks)

  * [1. Detection Model Validation (GPU)](#1-detection-model-validation-gpu)
  * [2. Detection Model Validation (DPU / Embedded FPGA)](#2-detection-model-validation-dpu--embedded-fpga)
  * [3. DPU Hardware Resource Consumption](#3-dpu-hardware-resource-consumption)
  * [4. GPU vs DPU Comparison](#4-gpu-vs-dpu-comparison)
  * [5. Multi-Object Tracking Results](#5-multi-object-tracking-results)
  * [6. Complete Detection + Tracking Pipeline](#6-complete-detection--tracking-pipeline)
* [Key Findings](#key-findings)
* [Limitations](#limitations)
* [Future Work](#future-work)
* [Contributing](#contributing)
* [Contact and Acknowledgements](#contact-and-acknowledgements)

---

## About The Project

The project focuses on the development of an autonomous embedded UAV defense and surveillance system capable of performing **real-time object detection and multi-object tracking**.

The system follows the **Tracking-by-Detection** paradigm:

1. A YOLO-based object detector processes each video frame.
2. Bounding boxes, class labels, and confidence scores are generated.
3. A tracking algorithm associates detections between consecutive frames.
4. A Kalman Filter predicts object motion.
5. Data association is performed using spatial information and, depending on the tracker, appearance information.
6. Each target is assigned and maintains a unique identity over time.

The final embedded implementation uses an **INT8-quantized YOLOv7-tiny model** deployed on the DPU of an **AMD Kria™ KV260 Vision AI Starter Kit**.

The final system achieved:

* **52 FPS** DPU inference
* **19 ms** YOLOv7-tiny inference latency
* **5.4 W** total power consumption
* **36.8 MB** used CMA memory
* **50.29% HOTA** with SORT
* **77.16% MOTP** with SORT
* Approximately **28.4 FPS** for the complete YOLOv7-tiny + SORT pipeline

These results demonstrate that lightweight deep learning models combined with FPGA-based acceleration can provide real-time AI processing under strict power and computational constraints.

---

## System Architecture

The system is based on a hardware/software co-design methodology.

![System Architecture](docs/Arquitetura.png)

The computationally intensive neural network inference is mapped to the DPU, while control, preprocessing, post-processing, and tracking operations are executed by the ARM processor.

The model deployment flow uses **AMD Vitis AI 3.5**, including calibration, INT8 quantization, compilation, and generation of the `.xmodel` representation required by the target DPU.

---

## Dataset


This project uses the **Military Assets Dataset**, containing **12 classes** of military-related objects and provided in YOLO format.

The dataset is publicly available on Kaggle:

**[Military Assets Dataset (12 Classes - YOLO8 Format)](https://www.kaggle.com/datasets/rawsi18/military-assets-dataset-12-classes-yolo8-format)**


### People

* Camouflaged soldiers
* Soldiers in standard uniforms
* Civilians

### Vehicles

* Military tanks
* Military trucks
* Aircraft
* Warships
* Civilian vehicles

### Weapons / Military Structures

* Firearms
* Heavy artillery
* Trenches

The dataset follows the standard YOLO directory structure:

```text
dataset/
├── train/
│   ├── images/
│   │   ├── 01.jpg
│   │   ├── 02.jpg
│   │   ├── 03.jpg
│   │   └── ...
│   │
│   └── labels/
│       ├── 01.txt
│       ├── 02.txt
│       ├── 03.txt
│       └── ...
│
├── val/
│   ├── images/
│   │   ├── 01.jpg
│   │   ├── 02.jpg
│   │   ├── 03.jpg
│   │   └── ...
│   │
│   └── labels/
│       ├── 01.txt
│       ├── 02.txt
│       ├── 03.txt
│       └── ...
│
├── test/
│   ├── images/
│   │   ├── 01.jpg
│   │   ├── 02.jpg
│   │   ├── 03.jpg
│   │   └── ...
│   │
│   └── labels/
│       ├── 01.txt
│       ├── 02.txt
│       ├── 03.txt
│       └── ...
│
└── military_dataset.yaml
```

Each image has a corresponding annotation file with the same filename:

```text
images/01.jpg  →  labels/01.txt
images/02.jpg  →  labels/02.txt
images/03.jpg  →  labels/03.txt
```

The dataset follows the YOLO annotation format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

For example:
```text
0 0.512 0.438 0.214 0.356
2 0.721 0.563 0.183 0.241
```

Each image is associated with a YOLO annotation file containing:

```text
<class_id> <center_x> <center_y> <width> <height>
```
The dataset was divided into training, validation, and testing subsets. The test set contains **1,396 images**.

---

## Built With

This project integrates the following frameworks, platforms, and toolchains:

* **Python**
* **PyTorch**
* **YOLOv7**
* **YOLOv8**
* **YOLO11**
* **OpenCV**
* **AMD Vitis AI 3.5**
* **Vitis AI Runtime (VART)**
* **INT8 Quantization**
* **AMD Kria™ KV260 Vision AI Starter Kit**
* **DPUCZDX8G**
* **ARM CPU**
* **FPGA / SoC**
* **SORT**
* **DeepSORT**
* **ByteTrack**
* **BoT-SORT**

---

## Getting Started

### Prerequisites

#### Host PC / Training Server

Recommended environment:

* Linux OS
* Ubuntu 20.04 / 22.04
* NVIDIA GPU with CUDA support
* Anaconda / Miniconda
* Docker
* PyTorch
* CUDA Toolkit

The training and model comparison experiments were performed using an NVIDIA GPU environment, including an NVIDIA GeForce RTX 4080.

#### Target Hardware

* AMD Kria™ KV260 Vision AI Starter Kit
* FPGA SoC
* ARM processor
* DPU
* Linux/PetaLinux-based environment
* Vitis AI Runtime (VART)

---

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/A46628/Detection_And_Tracking_Real_Time.git
cd Detection_And_Tracking_Real_Time
cd yolov7
```

2. **Create the Python environment:**

```bash
conda create -n drone-env python=3.8 -y
conda activate drone-env
```

3. **Install the dependencies:**

```bash
pip install -r requirements.txt
```

---

## Training

### Train YOLOv7-tiny

The YOLOv7-tiny model can be trained using:

```bash
python train.py \
    --workers 8 \
    --device 0 \
    --batch-size 16 \
    --data data/military_dataset.yaml \
    --img 640 640 \
    --cfg cfg/deploy/yolov7-tiny.yaml \
    --weights '' \
    --name yolov7-tiny-military
```

The best model is saved under:

```text
runs/train/yolov7-tiny-military/weights/best.pt
```

### Validate the trained model

```bash
python val.py \
    --weights runs/train/yolov7-tiny-military/weights/best.pt \
    --data data/military_dataset.yaml \
    --img 640 \
    --device 0
```

The validation process evaluates:

* Precision
* Recall
* mAP@0.5
* mAP@0.5:0.95

---

# Usage

## GPU Inference

### Single Image

```bash
python detect.py \
    --weights runs/train/yolov7-tiny-military/weights/best.pt \
    --conf 0.25 \
    --img-size 640 \
    --source path/to/image.jpg
```

### Video

```bash
python detect.py \
    --weights runs/train/yolov7-tiny-military/weights/best.pt \
    --conf 0.25 \
    --img-size 640 \
    --source path/to/video.mp4
```

### Webcam

```bash
python detect.py \
    --weights runs/train/yolov7-tiny-military/weights/best.pt \
    --conf 0.25 \
    --img-size 640 \
    --source 0
```



## Vitis AI Model Quantization and Compilation

The YOLOv7 model was prepared for deployment on the **AMD Kria™ KV260 DPU** using the **AMD Vitis AI** toolchain.

The quantization workflow was executed inside a Docker container based on the official **Xilinx Vitis AI CPU image**:

**[Xilinx Vitis AI CPU Docker Image](https://hub.docker.com/r/xilinx/vitis-ai-cpu)**

The complete workflow consists of three main stages:

```text
YOLOv7 FP32 Model
       │
       ▼
   Float Test
       │
       ▼
    Calibration
       │
       ▼
 INT8 Quantization
       │
       ▼
 Quantized Test
       │
       ▼
  Dump XModel
       │
       ▼
   .xmodel
       │
       ▼
    KV260 DPU
```

### 1. Create the Vitis AI Docker Container

First, pull the Vitis AI CPU image from Docker Hub:

```bash
docker pull xilinx/vitis-ai-cpu:latest
```

Create a Docker container with access to the project directory:

```bash
docker run -it \
    --name vitis-ai-yolov7 \
    --hostname vitis-ai-container \
    -v $(pwd):/workspace \
    xilinx/vitis-ai-cpu:latest \
    /bin/bash
```

### 2. Enter the YOLOv7 Project

Inside the Docker container:

```bash
cd /workspace/yolov7
```

The following commands assume that the YOLOv7 repository and trained weights are available inside this directory.

---

### 3. Floating-Point Model Validation

Before quantization, the original floating-point model is evaluated using the NNDCT pipeline.

```bash
cd /workspace/yolov7

python test_nndct.py \
    --data data/yolov7/custom_dataset_calib.yaml \
    --img 640 \
    --batch 1 \
    --conf 0.001 \
    --iou 0.65 \
    --device 0 \
    --weights yolov7.pt \
    --name yolov7_640_val \
    --quant_mode float
```

The `float` mode evaluates the original FP32 model and provides a baseline against which the quantized model can be compared.

---

### 4. Post-Training Quantization — Calibration

The next step performs **Post-Training Quantization (PTQ)** calibration.

```bash
cd /workspace/yolov7

python test_nndct.py \
    --data data/yolov7/custom_dataset_calib.yaml \
    --img 640 \
    --batch 1 \
    --conf 0.001 \
    --iou 0.65 \
    --device 0 \
    --weights yolov7.pt \
    --name yolov7_640_val \
    --quant_mode calib 
```

During calibration, representative data is used to determine the quantization parameters required to convert the model from floating-point representation to **INT8**.

---

### 5. INT8 Quantized Model Validation

After calibration, the quantized model can be evaluated using:

```bash
cd /workspace/yolov7

python test_nndct.py \
    --data data/yolov7/custom_dataset_calib.yaml \
    --img 640 \
    --batch 1 \
    --conf 0.001 \
    --iou 0.65 \
    --device 0 \
    --weights yolov7.pt \
    --name yolov7_640_val \
    --quant_mode test 
```

This stage verifies the accuracy of the quantized INT8 model before deployment to the DPU.

The resulting metrics can be compared against the original floating-point model to quantify the accuracy degradation caused by quantization.

---

### 6. Dump the Quantized Model

Finally, the calibrated and quantized model is exported using the `--dump_model` option:

```bash
cd /workspace/yolov7

python test_nndct.py \
    --data data/yolov7/custom_dataset_calib.yaml \
    --img 640 \
    --batch 1 \
    --conf 0.001 \
    --iou 0.65 \
    --device 0 \
    --weights yolov7.pt \
    --name yolov7_640_val \
    --quant_mode test \
    --dump_model
```
The dumped model is generated in a format suitable for the subsequent Vitis AI compilation flow.

The resulting `.xmodel` is then used for inference with the **DPU on the AMD Kria™ KV260**.

---



### 1. Download and Prepare the PetaLinux Image

The **PetaLinux** image used to initialize the AMD Kria™ KV260 was downloaded from the official AMD Embedded Software download portal.

**Official AMD Embedded Software Downloads:**

[AMD Embedded Software Downloads — PetaLinux](https://www.amd.com/en/support/downloads/adaptive-socs-and-fpgas/embedded-software.html?utm_source=chatgpt.com)

For this project, the **PetaLinux 2026.1** installer was used:

[PetaLinux 2026.1 Installer](https://account.amd.com/en/forms/downloads/xef.html?filename=petalinux-v2026.1-06061130-installer.run&utm_source=chatgpt.com)

The downloaded PetaLinux image was then transferred to a **MicroSD card** using **balenaEtcher**.

**balenaEtcher:**

[Download balenaEtcher](https://etcher.balena.io/?utm_source=chatgpt.com)

balenaEtcher provides a simple three-step process for writing an operating-system image to removable storage: **select the image, select the target drive, and flash the image**. It also validates the flashing process after completion.

#### MicroSD Preparation

The procedure used for the KV260 was:

```text id="j9p8xz"
PetaLinux 2026.1
       │
       │ Download from AMD
       ▼
PetaLinux Image
       │
       │ balenaEtcher
       ▼
   MicroSD Card
       │
       │ Insert into KV260
       ▼
AMD Kria™ KV260
       │
       ▼
   PetaLinux Boot
       │
       ▼
Embedded Linux Environment
```

The process is:

1. Download the appropriate **PetaLinux image** from the AMD Embedded Software portal.
2. Install and open **balenaEtcher**.
3. Insert the MicroSD card into the host computer.
4. Select the downloaded PetaLinux image.
5. Select the MicroSD card as the target device.
6. Flash the image to the MicroSD card.
7. Wait for the validation process to complete.
8. Safely eject the MicroSD card.
9. Insert the MicroSD card into the **AMD Kria™ KV260**.
10. Power on the board and allow PetaLinux to boot.

> **Warning:** Flashing an image with balenaEtcher erases the contents of the selected drive. Make sure the correct MicroSD card is selected before starting the flashing process.

### 2. Boot the KV260

After flashing, the MicroSD card is inserted into the KV260 and the board is powered on.

The PetaLinux environment provides the embedded Linux operating system required to run the application and interact with the KV260's processing system and programmable logic.

After booting, the board can be accessed through a local terminal or remotely through Ethernet/SSH:

```bash id="xj4p6m"
ssh <username>@<KV260_IP>
```

The Linux environment can then be verified with:

```bash id="7z2nqk"
uname -a
```

and:

```bash id="3f7y1a"
ls
```

Once the KV260 is successfully booted, the platform is ready for the next stages of the deployment process:

```text id="p2q7ks"
PetaLinux
    │
    ▼
KV260 Platform
    │
    ▼
Vitis AI Runtime
    │
    ▼
DPU
    │
    ▼
YOLOv7-tiny INT8
    │
    ▼
Object Detection
    │
    ▼
Tracking
```

> **Version note:** AMD currently identifies PetaLinux as a legacy/transitioning embedded Linux toolchain and recommends AMD Embedded Development Framework (EDF) for future projects. This project documents the PetaLinux-based setup used during development.





## Supported Trackers

### SORT

**Simple Online and Realtime Tracking**

SORT combines:

* Kalman Filter
* IoU-based matching
* Hungarian algorithm

It is computationally lightweight and particularly suitable for embedded platforms.

### DeepSORT

DeepSORT extends SORT by introducing a deep appearance descriptor.

It provides improved identity association under occlusion but introduces substantially higher computational requirements because an additional neural network is used for visual feature extraction.

### ByteTrack

ByteTrack associates both high-confidence and low-confidence detections.

The two-stage association strategy helps recover objects under:

* Partial occlusion
* Motion blur
* Low detector confidence

ByteTrack achieved the lowest number of identity switches in the experiments.

### BoT-SORT

BoT-SORT was studied as an advanced SORT-based approach incorporating camera-motion compensation and improved association strategies.

However, the final quantitative tracking comparison on the embedded platform focused on:

* SORT
* DeepSORT
* ByteTrack

---

# Results & Benchmarks

## 1. Detection Model Validation (GPU)

The following results were obtained during model validation on the military dataset.

| Model           |   mAP@0.5 | mAP@0.5:0.95 | Precision |    Recall |     FPS | Execution (ms) |
| --------------- | --------: | -----------: | --------: | --------: | ------: | -------------: |
| **YOLOv7-tiny** |     0.387 |        0.212 |     0.662 |     0.404 | **435** |        **2.3** |
| **YOLOv7**      | **0.629** |        0.424 |     0.683 | **0.610** |     278 |            3.6 |
| **YOLOv7x**     |     0.517 |        0.323 | **0.716** |     0.494 |     167 |            6.0 |
| **YOLOv8**      |     0.625 |    **0.448** |     0.621 |     0.578 |     294 |            3.4 |
| **YOLO11x**     |     0.610 |        0.439 |     0.618 |     0.572 |      97 |           10.3 |
| **YOLO11m**     |     0.620 |        0.455 |     0.630 |     0.583 |     232 |            4.3 |
| **YOLO11n**     |     0.560 |        0.375 |     0.695 |     0.502 |     370 |            2.7 |

YOLOv7 obtained the highest **mAP@0.5 (0.629)**, while YOLOv8 achieved the highest **mAP@0.5:0.95 (0.448)**.

YOLOv7-tiny provided the highest processing speed, reaching **435 FPS**, making it the most suitable candidate for embedded deployment despite its lower detection accuracy.

---

## 2. Detection Model Validation (DPU / Embedded FPGA)

After INT8 quantization and deployment to the KV260 DPU, the YOLOv7 variants achieved:

| Model           |   mAP@0.5 | mAP@0.5:0.95 | Precision |    Recall |    FPS |   Latency |
| --------------- | --------: | -----------: | --------: | --------: | -----: | --------: |
| **YOLOv7-tiny** | **0.387** |        0.198 |     0.587 |     0.376 | **52** | **19 ms** |
| **YOLOv7**      |     0.591 |    **0.356** |     0.647 | **0.583** |     10 |    104 ms |
| **YOLOv7x**     |     0.486 |        0.273 | **0.703** |     0.459 |      6 |    179 ms |

The results show a clear trade-off between model complexity and real-time performance.

**YOLOv7-tiny was selected as the final embedded model** because it was the only tested variant capable of maintaining a real-time processing rate while keeping power consumption low.

---

## 3. DPU Hardware Resource Consumption

Performance measurements on the AMD Kria™ KV260 were:

| Metric               | YOLOv7-tiny |   YOLOv7 |  YOLOv7x |
| -------------------- | ----------: | -------: | -------: |
| Total Power          |   **5.4 W** |   10.2 W |   10.9 W |
| Total Current        | **1068 mA** |  2020 mA |  2172 mA |
| Available CMA Memory |     1535 MB |  1453 MB |  1388 MB |
| Used CMA Memory      | **36.8 MB** | 116.9 MB | 179.8 MB |

YOLOv7-tiny required only **5.4 W** and **36.8 MB** of memory, making it the most suitable model for a battery-powered UAV platform.

---

## 4. GPU vs DPU Comparison

The embedded DPU provides significantly lower power consumption compared with the GPU environment.

| Metric       | Hardware | YOLOv7-tiny |   YOLOv7 |  YOLOv7x |
| ------------ | -------- | ----------: | -------: | -------: |
| mAP@0.5      | GPU      |       0.387 |    0.629 |    0.517 |
|              | DPU      |       0.387 |    0.591 |    0.486 |
| mAP@0.5:0.95 | GPU      |       0.212 |    0.424 |    0.323 |
|              | DPU      |       0.198 |    0.356 |    0.273 |
| Precision    | GPU      |       0.662 |    0.683 |    0.716 |
|              | DPU      |       0.587 |    0.647 |    0.703 |
| Recall       | GPU      |       0.404 |    0.610 |    0.494 |
|              | DPU      |       0.376 |    0.583 |    0.459 |
| FPS          | GPU      |         435 |      278 |      167 |
|              | DPU      |      **52** |       10 |        6 |
| Execution    | GPU      |      2.3 ms |   3.6 ms |   6.0 ms |
|              | DPU      |   **19 ms** |   104 ms |   179 ms |
| Power        | GPU      |      58.0 W |  134.0 W |  172.0 W |
|              | DPU      |   **5.4 W** |   10.2 W |   10.9 W |
| Memory       | GPU      |     376 MiB |  680 MiB |  966 MiB |
|              | DPU      | **36.8 MB** | 116.9 MB | 179.8 MB |

The DPU is slower than the GPU in raw inference throughput, but operates with dramatically lower power consumption.

For YOLOv7-tiny:

* GPU: **435 FPS / 58 W**
* DPU: **52 FPS / 5.4 W**

This represents a substantial reduction in power consumption while still satisfying the real-time requirement for the embedded application.

---

## 5. Multi-Object Tracking Results

The final tracking evaluation compared SORT, DeepSORT, and ByteTrack.

| Metric            |  ByteTrack | DeepSORT |       SORT |
| ----------------- | ---------: | -------: | ---------: |
| **HOTA**          |     32.26% |   22.43% | **50.29%** |
| **MOTA**          | **59.42%** |   38.51% |     50.01% |
| **MOTP**          |     53.29% |   36.00% | **77.16%** |
| False Negatives   |       1459 |  **669** |        738 |
| False Positives   |    **216** |     1844 |       1291 |
| Identity Switches |     **12** |       43 |         49 |
| Mostly Tracked    |         45 |  **112** |         97 |
| Mostly Lost       |        241 |  **157** |        170 |

### Tracking Analysis

**SORT** achieved the best spatial tracking performance:

* HOTA: **50.29%**
* MOTP: **77.16%**

Its low computational complexity makes it particularly suitable for the resource-constrained KV260.

**ByteTrack** achieved:

* MOTA: **59.42%**
* Lowest FP: **216**
* Lowest ID switches: **12**

Therefore, ByteTrack provides the best performance when identity consistency and false-positive reduction are the main priorities.

**DeepSORT** had the highest computational cost because its appearance extraction network runs on the host CPU. This caused a significant performance bottleneck in the embedded environment.

---

## 6. Complete Detection + Tracking Pipeline

The final pipeline combines YOLOv7-tiny with the tracking algorithms.

| Pipeline Stage        |    ByteTrack |   DeepSORT |         SORT |
| --------------------- | -----------: | ---------: | -----------: |
| Pre-processing (CPU)  |      9.86 ms |    9.86 ms |      9.86 ms |
| Inference (DPU)       |     19.01 ms |   19.01 ms |     19.01 ms |
| Post-processing (CPU) |      2.80 ms |    2.80 ms |      2.80 ms |
| Tracking (CPU)        |      4.02 ms | 2191.11 ms |  **3.50 ms** |
| **Total latency**     | **35.69 ms** | 2222.78 ms | **35.17 ms** |
| **FPS**               |     **28.0** |       0.45 |     **28.4** |

The complete system demonstrates that the DPU inference itself is not the only factor determining system-level performance.

For the final configuration:

```text
YOLOv7-tiny
     │
     ▼
INT8 DPU Inference
     │
     ├── 19.01 ms
     │
     ▼
Post-processing
     │
     ├── 2.80 ms
     │
     ▼
SORT
     │
     ├── 3.50 ms
     │
     ▼
Total ≈ 35.17 ms
     │
     ▼
≈ 28.4 FPS
```

The **YOLOv7-tiny + SORT** configuration achieved the best overall system-level balance between detection speed, tracking performance, latency, and embedded resource consumption.

---

## Key Findings

The main conclusions obtained from the experimental evaluation are:

### 1. YOLOv7-tiny is the best embedded detector

Although larger models achieved higher detection accuracy, YOLOv7-tiny provided the best balance between:

* Accuracy
* Inference speed
* Power consumption
* Memory footprint

### 2. INT8 quantization enables efficient FPGA deployment

The Vitis AI workflow enables the YOLOv7-tiny model to be converted from floating-point representation to **INT8**, reducing computational and memory requirements while allowing execution on the DPU.

### 3. The KV260 provides real-time AI at low power

The final YOLOv7-tiny DPU implementation achieved:

```text
52 FPS
19 ms inference latency
5.4 W power
36.8 MB memory
```

### 4. Tracking becomes an important system bottleneck

Although DPU inference reaches 52 FPS, the complete detection + tracking pipeline reaches approximately:

```text
28.4 FPS with SORT
28.0 FPS with ByteTrack
0.45 FPS with DeepSORT
```

This demonstrates that the performance of the complete system depends on both neural-network inference and CPU-based post-processing/tracking.

### 5. SORT is the preferred tracker for the KV260 configuration

SORT achieved:

```text
HOTA = 50.29%
MOTP = 77.16%
Total pipeline = 35.17 ms
FPS = 28.4
```

making it the best choice for the final resource-constrained implementation.

---

## Limitations

The current implementation has several limitations:

* The detector loses some accuracy after INT8 quantization.
* The DPU provides lower raw inference throughput than the GPU.
* CPU-based preprocessing and post-processing contribute significantly to total latency.
* DeepSORT is unsuitable for the current embedded configuration due to its high CPU computational cost.
* Tracking performance can degrade during severe occlusions, target proximity, motion blur, and target loss.
* The system was evaluated primarily using recorded video and controlled experimental scenarios rather than full autonomous flight tests.
* Navigation and autonomous flight control were not fully integrated into the final perception pipeline.

---

## Future Work

Future development can focus on:

* Hardware acceleration of preprocessing and post-processing.
* Optimization of tracking algorithms for FPGA/DPU execution.
* Integration of IMU and GPS information into the tracking pipeline.
* Integration with UAV navigation and flight-control systems.
* Camera-motion compensation for aerial tracking.
* Evaluation with larger and more diverse aerial datasets.
* Improved handling of severe occlusions and target re-identification.
* Exploration of newer lightweight YOLO architectures compatible with the DPU.
* Deployment and testing on a physical UAV platform.
* Integration of SLAM and autonomous navigation capabilities.
* Optimization of the full pipeline for higher end-to-end FPS.

---

## Contributing

Contributions, improvements, bug fixes, and experimental results are welcome.

To contribute:

1. Fork the repository.
2. Create a new branch:

```bash
git checkout -b feature/my-new-feature
```

3. Commit your changes:

```bash
git commit -m "Add new feature"
```

4. Push the branch:

```bash
git push origin feature/my-new-feature
```

5. Open a Pull Request.

---

## Contact and Acknowledgements

### Author

**Paulo Vitor Nunes Pereira Tavares**

MSc in Computer Engineering

ISEL — Instituto Superior de Engenharia de Lisboa

### Supervisor

**Dr. Mário Pereira Véstias**

ISEL

This project was developed as part of the Master's Project in Computer Engineering.

Special thanks to the project supervisor for the technical guidance, feedback, and support throughout the development of the project.

---

## Project Summary

| Component             | Final Configuration                         |
| --------------------- | ------------------------------------------- |
| Application           | UAV real-time object detection and tracking |
| Detection paradigm    | Tracking-by-Detection                       |
| Detection model       | **YOLOv7-tiny**                             |
| Model precision       | **INT8**                                    |
| Accelerator           | **DPU**                                     |
| Hardware              | **AMD Kria™ KV260 Vision AI Starter Kit**   |
| Dataset               | Military dataset — **12 classes**           |
| Detection FPS         | **52 FPS**                                  |
| DPU inference latency | **19 ms**                                   |
| Power consumption     | **5.4 W**                                   |
| Memory usage          | **36.8 MB**                                 |
| Final tracker         | **SORT**                                    |
| Tracking HOTA         | **50.29%**                                  |
| Tracking MOTP         | **77.16%**                                  |
| End-to-end FPS        | **28.4 FPS**                                |
| End-to-end latency    | **35.17 ms**                                |

---

## Citation

If you use this project or its results in academic work, please cite the corresponding Master's thesis:

```text
P. V. N. P. Tavares,
"Defense Drone System with Real-time Object Detection and Target Tracking,"
Master's Project, Instituto Superior de Engenharia de Lisboa (ISEL),
June 2026.
```

---

## License

This repository is intended for academic and research purposes.

Please check the repository configuration and individual dataset/model licenses before using the project or its components for commercial applications.
