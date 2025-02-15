# 🚀 Feature-Driven Data Communication: Drones & Cloud Server

**Optimizing Edge-to-Cloud Communication Through Feature Prioritization**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)


## 🌟 Core Innovation
**"Feature-Prioritized Transmission Strategy"** — The drone transmits only a 512-dimensional feature vector instead of the raw image:
- ✅ **Reduces bandwidth consumption by 60-70%**
- ⚡ **Maintains 85%+ classification accuracy (CIFAR-10)**
- 🔍 **Intelligent request mechanism**: Automatically requests the full image when confidence < 70%

## 🛠 Technical Highlights
- **Lightweight Feature Transmission**: 512D feature vector extracted by ResNet18
- **Dynamic Decision Engine**: Cloud-based adaptive request mechanism based on confidence
- **Edge-Cloud Collaborative Architecture**:
  - 🚁 **Drone Side**: Real-time feature extraction
  - ☁️ **Cloud Side**: Fast inference and decision-making

## ⚡ Quick Start
```bash
# Install dependencies
pip install torch torchvision
