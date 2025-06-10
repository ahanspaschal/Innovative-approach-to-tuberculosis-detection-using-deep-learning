# Deep Learning-Based Automatic Detection and Diagnosis of Tuberculosis from Chest X-ray Images
## 📌 Overview
Tuberculosis (TB) remains a leading infectious disease killer worldwide, with 10.6 million cases and 1.3 million deaths in 2022 (WHO). Traditional diagnostic methods (e.g., sputum microscopy, GeneXpert) are slow, expensive, and require specialized equipment—making them impractical in low-resource regions.

This project leverages deep learning (CNN models) to automate TB detection from chest X-rays, providing a fast, cost-effective, and scalable diagnostic solution. The best-performing model (DenseNet121) achieved 98% accuracy, outperforming existing approaches and demonstrating strong potential for real-world deployment.

## 🔍 Key Features & Technical Details
### 📂 Dataset
* Source: Publicly available medical imaging repositories
-<a href="https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset/data">Dataset</a>

* Size: 4,200 chest X-rays (CXR)

* 700 TB-positive (16.7%)

* 3,500 normal (83.3%)

* Format: DICOM/JPEG

* Resolution: 512×512 (resized to 224×224 for model input)
![chest](https://github.com/ahanspaschal/Innovative-approach-to-tuberculosis-detection-using-deep-learning/blob/main/chest.png)

### 🛠 Preprocessing Pipeline
* Resizing & Normalization

* Images resized to 224×224 pixels (compatible with CNN architectures).

* Pixel values scaled to [0, 1] for better convergence.

#### Class Imbalance Handling

* Applied class weighting to prevent bias toward the majority class (normal cases).
  
#### Train-Validation-Test Split

* 60% Training (2,520 images)

* 20% Validation (840 images)
* 20% Testing (840 images)

## 🤖 Deep Learning Models Evaluated
![models](https://github.com/ahanspaschal/Innovative-approach-to-tuberculosis-detection-using-deep-learning/blob/main/models.png)

### ⚙️ Training Configuration
* Optimizer: Adam (lr=0.001)

* Loss Function: Binary Cross-Entropy

* Batch Size: 32

* Epochs: 30

* Regularization: Dropout layers to prevent overfitting

## 📊 Results & Performance Comparison
#### 🏆 Best Model: DenseNet121
Metric	Normal (Class 0)	TB-Positive (Class 1)
Precision	0.99	0.92
Recall	0.98	0.97
F1-Score	0.99	0.94
Confusion Matrix (Test Set):

Predicted Normal	Predicted TB
Actual Normal	823	17
Actual TB	4	136

## 📈 Comparative Analysis
![comparative](https://github.com/ahanspaschal/Innovative-approach-to-tuberculosis-detection-using-deep-learning/blob/main/comparative.png)

## 💡 Key Insight:

* DenseNet121 outperformed others due to feature reuse via dense skip connections, crucial for detecting small TB opacities.

* ResNet50 failed on TB cases (only 4% recall) due to poor residual mapping during training.
![accuracy](https://github.com/ahanspaschal/Innovative-approach-to-tuberculosis-detection-using-deep-learning/blob/main/accuracy.png)

### 🌍 Real-World Impact & Applications
* ✅ Early Detection: Reduces diagnostic delays (critical for TB control).
* ✅ Cost-Effective: No need for expensive lab equipment.
* ✅ Scalable: Can be deployed in mobile clinics or telemedicine platforms.

Example Use Case:

Triage System: Prioritize high-risk patients for further testing.

Resource-Limited Settings: Deploy in rural areas with limited radiologists.

## 🚀 Future Work
Multi-Class Classification: Detect TB stages (early, advanced, cavitary).

Explainability: Add Grad-CAM visualizations to highlight infected regions.

Federated Learning: Improve model generalizability across diverse populations.

Real-World Testing: Partner with hospitals for clinical validation.

## 🛠 Technologies Used
Python (TensorFlow/Keras, OpenCV, Scikit-learn)

Google Colab (GPU acceleration)

Weights & Biases (W&B) (Experiment tracking)


## Thank You.
For more information or collaboration I can be reached via e-mail or linkedin.




