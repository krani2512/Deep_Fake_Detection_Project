# DeepFake AI Detection Framework
### Intelligent System for Identification and Authentication of Manipulated Visual Content

## 📌 Overview
Deepfakes are AI-generated or manipulated images and videos that can appear highly realistic and create serious challenges in media authenticity and digital security. Traditional CNN-based methods often struggle to detect subtle manipulations.

This project proposes a hybrid DeepFake detection framework using Capsule Networks and Siamese Networks to identify manipulated visual content. The Capsule Network captures spatial relationships and facial feature hierarchies, while the Siamese Network compares suspicious media with authentic references to detect inconsistencies.

The proposed model improves detection performance and enhances trust in digital media systems.

---

## 🎯 Objectives

- Detect manipulated images and videos accurately
- Capture structural inconsistencies in facial features
- Compare suspicious content with authentic references
- Improve media authenticity and digital trust
- Reduce false classifications in deepfake detection

---

## 🛠 Technologies Used

- Python
- TensorFlow
- OpenCV
- Deep Learning
- Capsule Networks
- Siamese Networks
- HTML / CSS / JavaScript
- Flask

---

## 📂 Dataset

The model uses the DeepFake Detection Challenge (DFDC) dataset from Kaggle.

Dataset Features:

- ~100,000 real videos
- ~100,000 fake videos
- Around 2000 subjects
- Resolution: 720×1280
- Format: MP4

Dataset Link:
https://www.kaggle.com/c/deepfake-detection-challenge

---

## ⚙ Methodology

### Step 1: Preprocessing
- Video frame extraction
- Face detection
- Image resizing and normalization
- Data augmentation

### Step 2: Capsule Network
- Extracts spatial relationships
- Detects structural inconsistencies

### Step 3: Siamese Network
- Compares image pairs
- Uses embedding similarity

### Step 4: Multi-Loss Training
- Capsule Margin Loss
- Contrastive Loss
- Triplet Loss

---

## 🏗 System Architecture

Input Video
↓
Frame Extraction
↓
Face Detection
↓
Preprocessing
↓
Capsule Network
↓
Siamese Network
↓
Feature Fusion
↓
Prediction (Real / Fake)

---

## 📊 Experimental Setup

- Learning Rate: 0.001
- Batch Size: 32
- Epochs: 50
- Optimizer: Adam
- Train : Validation : Test = 70:15:15

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|---------|----------|-----------|---------|-----------|------|
| CNN | 95.2% | 94.8% | 95.6% | 95.2% | 0.970 |
| Capsule-Siamese | 98.7% | 98.5% | 98.9% | 98.7% | 0.992 |

The proposed Capsule-Siamese model achieved superior performance compared to traditional CNN approaches.

---

## ✅ Advantages

- Better feature representation
- High detection accuracy
- Strong similarity learning
- Better generalization
- Robust against subtle manipulations

---

## ⚠ Limitations

- High computational complexity
- Requires larger datasets
- Increased training time

---

## 🔮 Future Work

- Audio + Video multimodal detection
- Transformer-based architecture
- Real-time deployment optimization
- Reference-free detection systems

---

## 👨‍💻 Authors

Kraniksa W  
Dhanavidhya J  
Hari Prabu A  
Vijay G  

Department of Artificial Intelligence and Data Science  
Muthayammal College of Engineering

---

## 📄 License

This project is developed for academic and educational purposes.
