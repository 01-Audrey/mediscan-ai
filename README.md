# 🏥 MediScan - AI-Powered Chest X-Ray Classifier

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/01-Audrey/mediscan-ai)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**MediScan** is an AI-powered web application that classifies chest X-rays as **NORMAL** or **PNEUMONIA** using deep learning and explainable AI techniques.

## 🚀 Live Demo

✨ **[Try MediScan Now on Hugging Face!](https://huggingface.co/spaces/01-Audrey/mediscan-ai)** ✨

Upload a chest X-ray and get instant AI-powered predictions with explainable Grad-CAM visualizations!

## 🎯 Features

- **🤖 Deep Learning Classification**: ResNet50 transfer learning model with 94.48% validation accuracy
- **🔍 Explainable AI**: Grad-CAM visualization shows exactly where the model is looking
- **📊 Confidence Scores**: Transparent probability scores for both classes
- **🎨 Professional UI**: Clean, medical-themed interface built with Gradio
- **⚡ Real-time Predictions**: Fast inference (~2-3 seconds)
- **🌐 Web-Based**: No installation required, accessible from any browser

## 📸 Screenshots

<img width="1795" height="942" alt="Screenshot 2025-11-16 233818" src="https://github.com/user-attachments/assets/e2f2e298-f889-470d-98bb-f19f6eeeb166" />

<img width="1478" height="1026" alt="Screenshot 2025-11-16 230907" src="https://github.com/user-attachments/assets/d5732621-65a0-4351-9ef9-6ed5c90542aa" />

### Live Application Interface

**Try it yourself:** [https://huggingface.co/spaces/01-Audrey/mediscan-ai](https://huggingface.co/spaces/01-Audrey/mediscan-ai)

### Features in Action:
- 🤖 **Real-time Classification:** Upload any chest X-ray for instant analysis
- 🎯 **Confidence Scores:** See probability distributions for both classes
- 🔍 **Grad-CAM Heatmaps:** Visual explanation of model's decision-making
- 📊 **Professional Interface:** Clean, medical-themed UI built with Gradio

## 🧠 How It Works

### 1. Model Architecture
- **Base Model**: ResNet50 pre-trained on ImageNet
- **Transfer Learning**: Fine-tuned on 5,863 chest X-ray images
- **Final Layer**: 2-class classifier (NORMAL vs PNEUMONIA)
- **Training**: 7 epochs with data augmentation

### 2. Grad-CAM Visualization
Gradient-weighted Class Activation Mapping (Grad-CAM) provides interpretability by:
- Computing gradients of the prediction with respect to feature maps
- Generating heatmaps showing regions of high attention
- Overlaying heatmaps on original images for visual explanation

### 3. Inference Pipeline
```
Input Image → Preprocessing → ResNet50 → Prediction + Grad-CAM → Visualization
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Validation Accuracy** | 94.48% |
| **Training Accuracy** | 92.78% |
| **Test Accuracy** | ~93% |
| **Parameters** | 25.5M (frozen: 25.5M, trainable: 4K) |

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch 2.1.0, torchvision
- **Web Framework**: Gradio 4.44.0
- **Image Processing**: PIL, OpenCV
- **Model**: ResNet50 Transfer Learning
- **Visualization**: Grad-CAM
- **Deployment**: Hugging Face Spaces

## 🌐 Deployment

MediScan is deployed on **Hugging Face Spaces** and accessible worldwide!

**Live URL:** [https://huggingface.co/spaces/01-Audrey/mediscan-ai](https://huggingface.co/spaces/01-Audrey/mediscan-ai)

**Deployment Stack:**
- Platform: Hugging Face Spaces
- Framework: Gradio 4.44.0
- Runtime: Python 3.10
- Hardware: CPU (free tier)
- Status: ✅ Live and Running

**Deployment Features:**
- ⚡ Fast inference (~2-3 seconds per image)
- 🌍 Globally accessible via public URL
- 🔄 Automatic scaling and load balancing
- 📦 Containerized deployment with Docker
- 🔒 Secure HTTPS connection

## 💻 Local Installation

### Prerequisites
- Python 3.10+
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/01-Audrey/mediscan-ai.git
cd mediscan-ai
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
python app.py
```

4. **Open in browser**
```
http://localhost:7860
```

## 📁 Project Structure
```
mediscan-ai/
├── app.py                  # Main Gradio application
├── models/
│   └── resnet50_best.pth  # Trained model weights (94.4 MB)
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── .gitignore           # Git ignore rules
```

## 🎓 Dataset

- **Source**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total Images**: 5,863
- **Classes**: NORMAL (1,583), PNEUMONIA (4,280)
- **Split**: 70% train, 15% validation, 15% test

## 🔬 Model Training Details

### Data Augmentation
- RandomResizedCrop (224x224)
- RandomHorizontalFlip (p=0.5)
- RandomRotation (±10°)
- ColorJitter (brightness/contrast ±0.2)
- ImageNet Normalization

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss**: CrossEntropyLoss
- **Scheduler**: ReduceLROnPlateau
- **Batch Size**: 32
- **Epochs**: 7
- **Device**: CPU/GPU compatible

## ⚠️ Medical Disclaimer

**IMPORTANT**: This is an **educational project** and **NOT** intended for clinical use. 

- Do NOT use for actual medical diagnosis
- Always consult qualified healthcare professionals
- This tool is for demonstration and learning purposes only
- The model has NOT been clinically validated

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Audrey**
- GitHub: [@01-Audrey](https://github.com/01-Audrey)
- Hugging Face: [@01-Audrey](https://huggingface.co/01-Audrey)
- LinkedIn: [Your LinkedIn]

## 🙏 Acknowledgments

- Dataset: Paul Mooney (Kaggle)
- ResNet Architecture: He et al. (2015)
- Grad-CAM: Selvaraju et al. (2017)
- Framework: Gradio Team
- Platform: Hugging Face

## 📚 References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
2. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. ICCV.
3. Kermany, D. S., et al. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. Cell.

---

⭐ **Star this repo** if you find it helpful!

🚀 **Try the live demo:** [https://huggingface.co/spaces/01-Audrey/mediscan-ai](https://huggingface.co/spaces/01-Audrey/mediscan-ai)

📧 **Questions?** Open an issue or reach out!
