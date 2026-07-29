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
