# 👤 Face Recognition System

A face recognition system built using OpenCV for detecting and recognizing faces from images.

---

## 🚀 Features

* Face detection using Haar Cascade
* Face recognition using LBPH algorithm
* Image preprocessing using grayscale conversion

---

## 🛠️ Tech Stack

* Python
* OpenCV
* NumPy

---

## ⚙️ Setup

```bash
pip install opencv-contrib-python numpy
```

---

## 📁 Dataset

Due to size limitations, the dataset is not included in this repository.  
You can use any labeled face dataset or create your own dataset in the following format:

dataset/
  person_1/
  person_2/
  person_3/

---

## 🧠 How It Works

* Images are converted to grayscale
* Faces are detected using Haar Cascade
* Face regions are extracted
* LBPH model is trained on labeled data
* Model predicts label and confidence for new images

---

## 👨‍💻 Author

**Palak Aggarwal**
