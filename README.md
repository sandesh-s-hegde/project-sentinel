# 👁️ Project Sentinel

### Empirical Grounding of Operational Risk

**Bridging Computer Vision and Real Options Analysis**

---

## 🎥 Dashboard

> **Operational volatility inferred directly from video-based flow entropy**
[Dashboard Preview](assets/project_sentinel_demo.mp4)

---

## 🚀 Abstract

Traditional operational risk models rely heavily on static historical data and ex-post assumptions. **Project Sentinel** introduces **Empirical Grounding**: the use of live video streams (e.g., shop floors, traffic junctions, logistics hubs) to extract **flow-level entropy** and dynamically calibrate financial risk parameters in real time.

This is not conventional object detection.
It is **financial modeling driven by pixel-level physics**.

By translating observed kinematic disorder into volatility estimates, Project Sentinel enables **real-time pricing of operational flexibility** using established financial theory.

---

## 💡 Key Features

* **Vision Engine**
  YOLOv8 combined with optical flow to track physical agents (people, vehicles, assets).

* **Entropy-Based Quantification**
  Converts spatiotemporal motion patterns into a **Shannon Entropy–based Flow Entropy score** (( \sigma_{flow} )).

* **Financial Valuation Layer**
  Empirically observed volatility feeds a **Black–Scholes–style Real Options model**, enabling continuous repricing of operational decisions.

* **Interactive Dashboard**
  Streamlit-based interface for live visualization of flows, entropy spikes, and implied volatility.

---

## 📄 Research Artefact (PDF)

📘 **Technical & Conceptual Artefact**

> Formal methodology, mathematical grounding, and validation results

👉 **View the PDF:**
[Project Sentinel – Technical Artefact (PDF)](assets/Project_Sentinel_Artefact.pdf)

---

## 📂 Included Assets

All validation and demonstration materials are located in the `assets/` directory:

* **`project_sentinel_demo.mp4`**
  Recorded Streamlit dashboard demonstrating real-time entropy and volatility dynamics.

* **`test_footage.mp4`**
  Industry-standard mixed-traffic video used for calibration.

* **`demo_result.jpg`**
  Validation output showing volatility spikes exceeding **25%** during vehicle ingress events.

* **`Project_Sentinel_Artefact.pdf`**
  Full technical artefact detailing theory, implementation, and empirical results.

---

## ⚙️ Methodology

1. **Ingestion**
   Raw video feeds are processed frame-by-frame.

2. **Perception**
   YOLOv8 detects agents; optical flow extracts motion vectors.

3. **Quantification**
   Motion distributions are converted into **normalized flow entropy**, mapped to an implied volatility index (( \sigma )).

4. **Valuation**
   The Black–Scholes formulation reprices the operational *real option* at one-second intervals.

---

## 🔄 System Integration

This module acts as the **Sensor Layer** in a broader **Financial Digital Twin**.

The extracted volatility metric (( \sigma_{flow} )) feeds into a downstream **Stochastic Inventory & Capacity Engine**, enabling:

* Risk-adjusted capacity planning
* Dynamic buffer allocation
* Automated operational decision-making

👉 **Related system:**
**Digital Capacity Optimizer**
[https://github.com/sandesh-s-hegde/digital_capacity_optimizer](https://github.com/sandesh-s-hegde/digital_capacity_optimizer)

---

## 📦 Installation

**Prerequisites:** Python 3.11

```bash
# 1. Clone the repository
git clone https://github.com/sandesh-s-hegde/project-sentinel.git
cd project_sentinel

# 2. Upgrade pip
py -m pip install --upgrade pip

# 3. Install PyTorch (Windows-optimized)
py -m pip install torch==2.5.1 torchvision==0.20.1

# 4. Install tracking dependency
py -m pip install lapx

# 5. Install remaining requirements
py -m pip install -r requirements.txt
```

---

## 🧠 Conceptual Positioning

Project Sentinel operates at the intersection of:

* Computer Vision
* Information Theory
* Financial Engineering
* Operations & Risk Management

It reframes **physical uncertainty as a financial signal**, enabling a new class of empirically grounded operational decision systems.