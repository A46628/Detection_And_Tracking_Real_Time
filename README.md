Markdown
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
   git clone [https://github.com/seu-usuario/Defense-Drone-System.git](https://github.com/seu-usuario/Defense-Drone-System.git)
   cd Defense-Drone-System
Set Up Python Environment:

Bash
conda create -n drone-env python=3.8 -y
conda activate drone-env
pip install -r requirements.txt
Train / Validate Model Locally:
To train the detection model on the custom military dataset:

Bash
python train.py --workers 8 --device 0 --batch-size 16 --data data/military_dataset.yaml --img 640 640 --cfg cfg/deploy/yolov7-tiny.yaml --weights '' --name yolov7-tiny-military
Quantize and Compile for DPU (Vitis-AI):
Enter the Vitis-AI Docker container and run the calibration/compilation scripts:

Bash
# Quantization (INT8)
python quantize.py --model_path yolov7-tiny.pt --output_dir ./quantized_model

# Compilation for KV260 DPU
vai_c_xir -x ./quantized_model/YOLOv7_tiny_org.xmodel -a /arch/DPUCCZDX8G/KV260/arch.json -o ./compiled_model -n yolov7_tiny_kv260
Deploy to Kria KV260:
Copy the generated .xmodel file along with the application runtime code to your Kria board:

Bash
scp ./compiled_model/yolov7_tiny_kv260.xmodel petalinux@<board-ip>:/home/petalinux/
Usage
Run the real-time detection and tracking pipeline on the Kria KV260 board:

Bash
python main_tracking.py --model yolov7_tiny_kv260.xmodel --tracker sort --video input_sample.mp4
Tracking-by-Detection Pipeline
The system processes video input sequentially:

Pre-processing: Video frame acquisition, resizing (640×640), and input normalization.

Inference (Hardware Accelerated): YOLO detection running on the DPU overlay.

Post-processing: Dequantization, Sigmoid calculation, Non-Maximum Suppression (NMS), and confidence filtering executed on the host ARM CPU.

Data Association: Detections are assigned to existing trajectories via Kalman Filter estimation and the Hungarian algorithm.

+------------------+     +-------------------+     +--------------------+     +-------------------+
|  Camera / Video  | --> | YOLOv7-tiny (DPU) | --> | Post-Process (NMS) | --> |  Multi-Object     |
|   Frame Input    |     |   INT8 Inference  |     |     (ARM CPU)      |     |  Tracker (SORT)   |
+------------------+     +-------------------+     +--------------------+     +-------------------+
Supported Trackers
SORT: High frame-rate tracking relying on IoU and linear motion models.

ByteTrack: Utilizes low-confidence detections to recover partially hidden objects.

DeepSORT: Integrates visual appearance descriptors to handle prolonged occlusions.

Results & Benchmarks
1. Detection Model Validation (GPU)
Performance & Accuracy
Model	mAP 
50
​
 	mAP 
50−95
​
 	P	R	FPS	Exec. (ms)
YOLOv7x	0.517	0.323	0.716	0.494	167	6.0
YOLOv7-tiny	0.387	0.212	0.662	0.404	435	2.3
YOLOv7	0.629	0.424	0.683	0.610	278	3.6
YOLOv8	0.625	0.448	0.621	0.578	294	3.4
YOLO11x	0.610	0.439	0.618	0.572	97	10.3
YOLO11m	0.620	0.455	0.630	0.583	232	4.3
YOLO11n	0.560	0.375	0.695	0.502	370	2.7
Hardware Resource Consumption
Metric	YOLOv7-tiny	YOLOv7	YOLOv7-x
Parameters	6.03 Million	36.54 Million	70.85 Million
GFLOPS	13.1	103.3	188.2
Power Consumption	58 W	134 W	172 W
GPU Memory Usage (VRAM)	376 MiB	680 MiB	966 MiB
GPU Utilization	31%	54%	64%
2. Detection Model Validation (DPU / Embedded FPGA)
Model Performance on DPU (INT8 Quantized)
Modelo	mAP@0.5	mAP@0.5:0.95	P	R	FPS	Speed (ms)
YOLOv7-tiny	0.387	0.198	0.587	0.376	52	19.00
YOLOv7	0.591	0.356	0.647	0.583	10	104.00
YOLOv7x	0.486	0.273	0.703	0.459	6	179.00
Power & Memory Profile (Kria KV260 Target Hardware)
Metric	YOLOv7-tiny	YOLOv7 (Base)	YOLOv7-x
Total Power Consumption	5.4 W	10.2 W	10.9 W
Total Current	1068 mA	2020 mA	2172 mA
Available CMA Memory	1535 MB	1453 MB	1388 MB
Used CMA Memory	36.8 MB	116.9 MB	179.8 MB
3. DPU Latency & Pipeline Breakdown
Optimized Execution Breakdown (YOLOv7-Tiny)
Pipeline Stage	Processing Time (ms)
Pre-processing (CPU)	9.22
Inference (DPU)	19.06
Post-processing (CPU)	2.05
Total Time (T 
total
​
 )	30.16
Achieved FPS	33.16
Network Output Grid Scale
Scale	Dimension	Quantity
Small Objects	80×80 Grid	6,400 cells
Medium Objects	40×40 Grid	1,600 cells
Large Objects	20×20 Grid	400 cells
Total Candidate Positions (N)	Sum of 3 scales	8,400 cells
Attributes per Cell (C)	3 Anchors × (4 BBox + 1 Conf + 12 Classes)	51 channels
Data Volume	8,400×51 (INT8)	428,400 (INT8)
4. System Results (Detection + Tracking)
Tracking Metrics Comparison
Metric	ByteTrack	DeepSORT	SORT
HOTA	32.26%	22.43%	50.29%
MOTA	59.42%	38.51%	50.01%
MOTP	53.29%	36.00%	77.16%
FN	1459	669	738
FP	216	1844	1291
IDSW	12	43	49
End-to-End Pipeline Latency Comparison
Pipeline	YOLOv7-Tiny + ByteTrack (ms)	YOLOv7-Tiny + DeepSORT (ms)	YOLOv7-Tiny + SORT (ms)
Pre-processing (CPU)	9.86	9.86	9.86
Inference (DPU)	19.01	19.01	19.01
Post-processing (CPU)	2.80	2.80	2.80
Tracking (CPU)	4.02	2191.11	3.50
Total Time (T 
total
​
 )	35.69	2222.78	35.17
Overall FPS	28.0	0.45	28.4
Contributing
Contributions are welcome! If you'd like to improve tracking performance, add new feature modules, or optimize hardware deployment:

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

Contact and Acknowledgements
Author: Paulo Vitor Nunes Pereira Tavares

Advisor: Dr. Mário Pereira Véstias

Institution: Instituto Superior de Engenharia de Lisboa (ISEL)

Project developed as part of the Master's Dissertation in Computer Science and Engineering:

Defense Drone System with Real-time Object Detection and Target Tracking (June 2026).

Project Link: https://github.com/seu-usuario/Defense-Drone-System
