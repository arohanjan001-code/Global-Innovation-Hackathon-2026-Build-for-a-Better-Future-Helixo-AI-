# Helixo AI - V2 Architecture & Upgrade Plan

*This document summarizes all architectural decisions and hardware upgrades discussed, serving as the blueprint for our next development phase.*

## 1. Core Architecture Shift (Classification to Object Detection)
- **Decision:** Moving from basic Image Classification (MobileNetV2 classifier) to **Object Detection** (Using YOLO-nano / MobileNet-SSD architectures).
- **The "Why":** Object Detection drastically improves accuracy by drawing bounding boxes around the "Head/Helmet" and ignoring the background. 
- **The Error Handling:** This natively solves the "blocked camera" or "random object" glitch. If no bounding box is drawn at all (camera blocked, empty road), the system instantly triggers an `Error / Clean Camera` state.
- **Development Strategy:** We will NOT build neural networks from scratch. We will leverage industry-standard open-source architectures (like YOLO/MobileNet) and train them exclusively on our custom Helixo dataset.

## 2. Hardware Selection (Dedicated NPU under ₹2000)
To run Object Detection smoothly (ESP32 struggles with this), we are shifting to chips with dedicated AI accelerators (NPUs).
- **Primary Choice: Kendryte K210 (Sipeed Maix)**
  - **Reason:** Built for real-time edge AI object detection at high frame rates. It has an "Instant Boot" advantage (bare-metal) which is critical for syncing with the bike's ignition switch.
- **Alternative: Luckfox Pico (RV1103)**
  - **Reason:** Very high AI power for cheap. However, it runs a micro-Linux OS which takes 10-15 seconds to boot up upon key turn.

## 3. Camera Module Strategy
- **The Lens:** Upgrading from the generic 2MP (OV2640) to **OV5640 (5 Megapixel)** for far superior dynamic range, sharpness, and low-light performance.
- **The Pro-Tip:** The OV5640 must have a **Wide Angle Lens (120° - 160°)**. Because the bike dashboard is very close to the rider's chest/face, a standard lens acts "zoomed in". A wide-angle lens ensures the entire helmet frame is visible regardless of seating position.

## 4. Product Claims & Marketing
- As a business and legal safeguard, the product will **never claim 100% accuracy**.
- It will be marketed as an advanced AI hardware system with **"98%+ Precision with Smart Built-in Failsafes"**. This covers extreme edge cases like mud on the lens or torrential rain.

---

## 🚀 Execution Plan (Next Steps for Tomorrow)
When development resumes, we will follow these exact steps sequentially:

### Step 1: Bounding Box Annotation
We will start the "dumb-work". Our existing 1100+ augmented dataset images need to be annotated (drawing rectangles manually over every helmet/head) using a tool like LabelImg, MakeSense.ai, or Roboflow.

### Step 2: Object Detection Pipeline Setup
We will write a new python notebook/script to train an Object Detection model (instead of our old classifier). We will feed it the newly annotated bounding-box dataset.

### Step 3: Hardware Firmware Logic (C++)
Once the new model (`.tflite` or K210 format) is generated, we will write the hardware logic:
1. `If (BoundingBox == Helmet)` ➔ Enable Ignition Relay.
2. `If (BoundingBox == No_Helmet)` ➔ Cut Ignition Relay.
3. `If (No BoundingBox Detected)` ➔ Trigger "Clean Camera / Out of Frame" Error State.
