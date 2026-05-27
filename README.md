<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1f6feb,100:0d1117&height=140&section=header&text=Mars%20Surface%20Landmark%20Classifier&fontColor=58a6ff&fontSize=28&fontAlign=50&fontAlignY=45&desc=CNN%20%7C%20NASA%20HiRISE%20%7C%20TensorFlow&descColor=8b949e&descSize=14&descAlignY=68" width="100%"/>

</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-0d1117?style=flat-square&logo=python&logoColor=58a6ff)
![TensorFlow](https://img.shields.io/badge/TensorFlow-0d1117?style=flat-square&logo=tensorflow&logoColor=58a6ff)
![Keras](https://img.shields.io/badge/Keras-0d1117?style=flat-square&logo=keras&logoColor=58a6ff)
![NumPy](https://img.shields.io/badge/NumPy-0d1117?style=flat-square&logo=numpy&logoColor=58a6ff)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0d1117?style=flat-square&logo=scikitlearn&logoColor=58a6ff)

</div>

---

## Overview

Convolutional Neural Network trained on NASA's **HiRISE map-proj-v3** dataset to classify 8 geologically distinct Mars surface landmark types from orbital imagery.

---

## Classes

| ID | Label | ID | Label |
|---|---|---|---|
| 0 | other | 4 | bright dune |
| 1 | crater | 5 | impact ejecta |
| 2 | dark dune | 6 | swiss cheese |
| 3 | slope streak | 7 | spider |

---

## Data Pipeline

```
labels-map-proj-v3.txt
        ↓
  parse labels → CSV
        ↓
  stratified 80/20 train/test split
        ↓
  tf.data pipeline
        ↓
  preprocess (128×128, normalise to [0,1])
        ↓
  augmentation → batch(32) → prefetch
        ↓
  CNN training
```

**Augmentation:**

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
])
```

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│  input_layer     (128, 128, 3)                   │
├──────────────────────────────────────────────────┤
│  data_augmentation                               │
├──────────────────────────────────────────────────┤
│  conv1           Conv2D(32, 3×3, relu, same)     │
│  pool1           MaxPooling2D(2×2)               │
├──────────────────────────────────────────────────┤
│  conv2           Conv2D(64, 3×3, relu, same)     │
│                  + L2(1e-4)                      │
│  pool2           MaxPooling2D(2×2)               │
├──────────────────────────────────────────────────┤
│  conv3           Conv2D(128, 3×3, relu, same)    │
│                  + L2(1e-4)                      │
│  pool3           MaxPooling2D(2×2)               │
├──────────────────────────────────────────────────┤
│  flatten                                         │
│  fc1             Dense(256, relu) + L2(1e-4)     │
│  dropout1        Dropout(0.3)                    │
├──────────────────────────────────────────────────┤
│  output          Dense(8, softmax)               │
└──────────────────────────────────────────────────┘
```

---

## Training Config

```
Optimizer     Adam  (lr = 0.001)
Loss          Sparse Categorical Crossentropy
Metric        Accuracy
Epochs        18 (max)
Early stop    patience = 5  |  monitor = val_loss
              restore_best_weights = True
Batch size    32
Image size    128 × 128
Split         80% train  /  20% test  (stratified)
```

---

## Setup

```bash
git clone https://github.com/ayushcompsx/mars-landmark-classifier
cd mars-landmark-classifier
pip install -r requirements.txt
```

Download the dataset from [NASA AI4Mars HiRISE](https://data.nasa.gov/Space-Science/AI4Mars/cawu-k4s7) and place:
- `map-proj-v3/` — image directory
- `labels-map-proj-v3.txt` — label file

```bash
jupyter notebook mars_landmark_classifier.ipynb
```

---

## Stack

`TensorFlow` `Keras` `NumPy` `pandas` `scikit-learn` `Matplotlib` `Pillow`

---

<div align="center">

*MSc AI & Machine Learning · Queen Mary University of London*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1f6feb,100:0d1117&height=80&section=footer" width="100%"/>

</div>
