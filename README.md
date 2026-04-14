# 🌿 CNN Plant Disease Detection

A deep learning project that uses a Convolutional Neural Network (CNN) to classify plant diseases from leaf images using the PlantVillage dataset.

---

## 📌 Project Overview

Plant diseases can significantly affect agricultural productivity. This project leverages deep learning to automatically detect and classify plant diseases from images, helping in early diagnosis and better crop management.

This model is built using **PyTorch** and trained on a multi-class dataset containing various plant diseases.

---

## 🚀 Features

* Multi-class plant disease classification (38 classes)
* Custom PyTorch Dataset implementation
* Image preprocessing and normalization
* CNN architecture with:

  * Convolutional layers
  * Batch Normalization
  * Max Pooling
  * Dropout for regularization
* Training and evaluation pipeline
* Test accuracy calculation

---

## 🧠 Model Architecture

The CNN model consists of:

* 3 Convolutional Blocks:

  * Conv2D → ReLU → BatchNorm → MaxPool
* Fully Connected Layers:

  * Flatten → Dense → ReLU → Dropout
  * Output layer for classification

---

## 🗂️ Dataset

* Dataset: PlantVillage
* Source: Kaggle
* Classes: 38 plant disease categories
* Images: 50,000+ leaf images

---

## ⚙️ Technologies Used

* Python
* PyTorch
* Torchvision
* PIL (Python Imaging Library)
* KaggleHub

---

## 🏗️ Project Structure

```
cnn-plant-disease-detection/
│
├── cnn_project.py
├── README.md
└── requirements.txt (optional)
```

---

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/cnn-plant-disease-detection.git
cd cnn-plant-disease-detection
```

Install dependencies:

```bash
pip install torch torchvision pillow kagglehub
```

---

## ▶️ How to Run

```bash
python cnn_project.py
```

---

## 📊 Evaluation

The model includes an evaluation function to calculate test accuracy:

```python
test_acc = evaluate(test_loader)
print("Test Accuracy:", test_acc)
```

---

## 📈 Results

* Achieved high accuracy on test dataset
* Model performs well on multi-class classification tasks

---

## 💡 Future Improvements

* Train on full dataset (instead of subset)
* Add data augmentation
* Use transfer learning (ResNet, EfficientNet)
* Deploy as a web application
* Add real-time plant disease detection

---

## 👨‍💻 Author

* Fahim Muntasir Tuhin (Data Science Student)

---

## ⭐ Contribute

Feel free to fork this repository and improve the project.

---

## 📜 License

This project is for educational purposes.
