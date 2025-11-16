\# 🏥 MediScan - AI-Powered Chest X-Ray Classifier



\[!\[Streamlit App](https://static.streamlit.io/badges/streamlit\_badge\_black\_white.svg)](https://your-app-url-here.streamlit.app)

\[!\[Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)

\[!\[PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



\*\*MediScan\*\* is an AI-powered web application that classifies chest X-rays as \*\*NORMAL\*\* or \*\*PNEUMONIA\*\* using deep learning and explainable AI techniques.



\## 🎯 Features



\- \*\*🤖 Deep Learning Classification\*\*: ResNet50 transfer learning model with 94.48% validation accuracy

\- \*\*🔍 Explainable AI\*\*: Grad-CAM visualization shows exactly where the model is looking

\- \*\*📊 Confidence Scores\*\*: Transparent probability scores for both classes

\- \*\*🎨 Professional UI\*\*: Clean, medical-themed interface built with Streamlit

\- \*\*⚡ Real-time Predictions\*\*: Fast inference (~2-3 seconds)

\- \*\*🌐 Web-Based\*\*: No installation required, accessible from any browser



\## 🚀 Live Demo



Try the app here: \*\*\[MediScan Live Demo](https://your-app-url-here.streamlit.app)\*\* \*(coming soon)\*



\## 📸 Screenshots



\### Normal X-Ray Classification

!\[Normal Prediction](screenshots/normal\_prediction.png)

\*MediScan correctly identifies healthy lungs with 94.84% confidence\*



\### Pneumonia Detection

!\[Pneumonia Prediction](screenshots/pneumonia\_prediction.png)

\*Model detects pneumonia with 99.47% confidence, with Grad-CAM highlighting affected areas\*



\## 🧠 How It Works



\### 1. Model Architecture

\- \*\*Base Model\*\*: ResNet50 pre-trained on ImageNet

\- \*\*Transfer Learning\*\*: Fine-tuned on 5,863 chest X-ray images

\- \*\*Final Layer\*\*: 2-class classifier (NORMAL vs PNEUMONIA)

\- \*\*Training\*\*: 7 epochs with data augmentation



\### 2. Grad-CAM Visualization

Gradient-weighted Class Activation Mapping (Grad-CAM) provides interpretability by:

\- Computing gradients of the prediction with respect to feature maps

\- Generating heatmaps showing regions of high attention

\- Overlaying heatmaps on original images for visual explanation



\### 3. Inference Pipeline

```

Input Image → Preprocessing → ResNet50 → Prediction + Grad-CAM → Visualization

```



\## 📊 Model Performance



| Metric | Score |

|--------|-------|

| \*\*Validation Accuracy\*\* | 94.48% |

| \*\*Training Accuracy\*\* | 92.78% |

| \*\*Test Accuracy\*\* | ~93% |

| \*\*Parameters\*\* | 25.5M (frozen: 25.5M, trainable: 4K) |



\## 🛠️ Technology Stack



\- \*\*Deep Learning\*\*: PyTorch 2.5.1, torchvision

\- \*\*Web Framework\*\*: Streamlit 1.51.0

\- \*\*Image Processing\*\*: PIL, OpenCV

\- \*\*Model\*\*: ResNet50 Transfer Learning

\- \*\*Visualization\*\*: Grad-CAM



\## 💻 Local Installation



\### Prerequisites

\- Python 3.9+

\- pip package manager



\### Setup



1\. \*\*Clone the repository\*\*

```bash

git clone https://github.com/01-Audrey/mediscan-ai.git

cd mediscan-ai

```



2\. \*\*Install dependencies\*\*

```bash

pip install -r requirements.txt

```



3\. \*\*Run the app\*\*

```bash

streamlit run app.py

```



4\. \*\*Open in browser\*\*

```

http://localhost:8501

```



\## 📁 Project Structure

```

mediscan-ai/

├── app.py                  # Main Streamlit application

├── models/

│   └── resnet50\_best.pth  # Trained model weights

├── .streamlit/

│   └── config.toml        # Streamlit configuration

├── requirements.txt       # Python dependencies

├── README.md             # Project documentation

└── .gitignore           # Git ignore rules

```



\## 🎓 Dataset



\- \*\*Source\*\*: \[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

\- \*\*Total Images\*\*: 5,863

\- \*\*Classes\*\*: NORMAL (1,583), PNEUMONIA (4,280)

\- \*\*Split\*\*: 70% train, 15% validation, 15% test



\## 🔬 Model Training Details



\### Data Augmentation

\- RandomResizedCrop (224x224)

\- RandomHorizontalFlip (p=0.5)

\- RandomRotation (±10°)

\- ColorJitter (brightness/contrast ±0.2)

\- ImageNet Normalization



\### Training Configuration

\- \*\*Optimizer\*\*: Adam (lr=0.001)

\- \*\*Loss\*\*: CrossEntropyLoss

\- \*\*Scheduler\*\*: ReduceLROnPlateau

\- \*\*Batch Size\*\*: 32

\- \*\*Epochs\*\*: 7

\- \*\*Device\*\*: CPU/GPU compatible



\## ⚠️ Medical Disclaimer



\*\*IMPORTANT\*\*: This is an \*\*educational project\*\* and \*\*NOT\*\* intended for clinical use. 



\- Do NOT use for actual medical diagnosis

\- Always consult qualified healthcare professionals

\- This tool is for demonstration and learning purposes only

\- The model has NOT been clinically validated



\## 🤝 Contributing



Contributions are welcome! Please feel free to submit a Pull Request.



1\. Fork the repository

2\. Create your feature branch (`git checkout -b feature/AmazingFeature`)

3\. Commit your changes (`git commit -m 'Add some AmazingFeature'`)

4\. Push to the branch (`git push origin feature/AmazingFeature`)

5\. Open a Pull Request



\## 📝 License



This project is licensed under the MIT License - see the \[LICENSE](LICENSE) file for details.



\## 👤 Author



\*\*Audrey\*\*

\- GitHub: \[@01-Audrey](https://github.com/01-Audrey)

\- Portfolio: \[Your Portfolio URL]

\- LinkedIn: \[Your LinkedIn]



\## 🙏 Acknowledgments



\- Dataset: Paul Mooney (Kaggle)

\- ResNet Architecture: He et al. (2015)

\- Grad-CAM: Selvaraju et al. (2017)

\- Framework: Streamlit Team



\## 📚 References



1\. He, K., Zhang, X., Ren, S., \& Sun, J. (2016). Deep residual learning for image recognition. CVPR.

2\. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. ICCV.

3\. Kermany, D. S., et al. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. Cell.



---



⭐ \*\*Star this repo\*\* if you find it helpful!



📧 \*\*Questions?\*\* Open an issue or reach out!

