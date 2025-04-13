# ReviveAI ✨


<p align="center">
  <img src="./assets/revive banner.png" alt="ReviveAI Logo" width="50%"/>
<p align="center">
  <em>Restore your memories. AI-powered image deblurring, sharpening, and scratch removal.</em>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Build-Passing-brightgreen" alt="Build Status"></a>
  <a href="link/to/your/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="License"></a> 
  <a href="#"><img src="https://img.shields.io/badge/Python-3.8+-blueviolet" alt="Python Version"></a>
  <a href="link/to/your/contributing/guide"><img src="https://img.shields.io/badge/Contributions-Welcome-orange" alt="Contributions Welcome"></a>
</p>

---

## 📖 About ReviveAI

ReviveAI leverages the power of Artificial Intelligence to breathe new life into your old or degraded photographs. Whether it's blurriness from camera shake, general lack of sharpness, or physical damage like scratches, ReviveAI aims to restore clarity and detail, preserving your precious moments.

This project utilizes state-of-the-art deep learning models trained specifically for image restoration tasks. Our goal is to provide an accessible tool for enhancing image quality significantly.

---

## 🔥 Key Features
<p align = "center"><img src="./assets/features.png" alt="Features" width="100%" align = "center"/></p>

*   **✅ Completed - Image Sharpening:** Enhances fine details and edges for a crisper look.
*   **✅ Completed - Scratch Removal:** Intelligently detects and inpaints scratches and minor damages on photographs.
*   **🛠️ Work-in-progress - Image Colorization(Coming Soon):** Adds realistic color to grayscale images.

---

## ✨ Before & After Showcase

See the results of ReviveAI in action!

<p align="center"> <!-- Center align the entire table block -->

| Examples                                    | Task Performed     |
| :-----------------------------------------: | :----------------- |
| <img src="./assets/sharpen1.png" alt="ReviveAI Sharp Result 1" width="650"> | Image Sharpening   |
| <img src="./assets/sharpen2.png" alt="ReviveAI Sharp Result 2" width="650"> | Image Sharpening   |
| <img src="./assets/scratch1.png" alt="ReviveAI Scratch Removal Result 1" width="650"> | Scratch Removal    |
| <img src="./assets/scratch2.png" alt="ReviveAI Scratch Removal Result 2" width="650"> | Scratch Removal    |
</p> <!-- End of center alignment -->

---


## 🛠️ Tech Stack

The project is built using the following technologies:

<p align="left"> <!-- Or use align="center" if you prefer -->
  <a href="https://www.python.org/" target="_blank">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python"/>
  </a> 
  <a href="https://www.tensorflow.org/" target="_blank">
    <img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" alt="TensorFlow"/>
  </a> 
  <a href="https://opencv.org/" target="_blank">
    <img src="https://img.shields.io/badge/opencv-%235C3EE8.svg?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
  </a> 
  <a href="https://numpy.org/" target="_blank">
    <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  </a> 
  <!-- Add more badges here if needed for GUI, Web Frameworks, etc. -->
</p>

---


## 📊 Implementation Status

Track the development progress of ReviveAI's key features and components:

| Feature / Component          | Status                   | Notes / Remarks (Optional) |
| :--------------------------- | :----------------------- | :------------------------- |
| Image Deblurring/Sharpening  | ✅ Completed             | Core model functional      |
| Scratch Removal              | ✅ Completed             | Core model functional      |
| Image Colorization           | 🚧 In Progress           | Model integration underway |
| Website Design (UI/UX)       | ✅ Completed             | Design finalized           |
| Website Implementation       | ⌛ Pending               | Backend/Frontend dev needed|
| Website Deployment           | ⌛ Pending               | Requires server setup      |

---

## 🚀 Getting Started

Follow these steps to get ReviveAI running on your local machine or in a Jupyter/Kaggle notebook.

### 1. Prerequisites

Ensure you have the following installed:

- Python 3.8 or above  
- `pip` (Python package manager)  
- Git (for cloning the repository)  
- [Hugging Face CLI (optional)](https://huggingface.co/docs/huggingface_hub/quick-start)  
- Jupyter Notebook or run on [Kaggle](https://kaggle.com) / [Google Colab](https://colab.research.google.com)


---

### 2. Clone the Repository

```bash
git clone https://github.com/Zummya/ReviveAI.git
cd ReviveAI
```

---

### 3. Set Up the Environment

We recommend using a virtual environment:

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

---

## 🎯 Load Pretrained Models

All models are hosted on the Hugging Face Hub for convenience and version control.

### 🔹 Load Image Sharpening Model

```python
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model

model_path = hf_hub_download(
    repo_id="Sami-on-hugging-face/RevAI_Deblur_Model", 
    filename="SharpeningModel_512_30Epochs.keras"
)
model = load_model(model_path, compile=False)
```

---

### 🔹 Load Scratch Removal Model

```python
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model

model_path = hf_hub_download(
    repo_id="Sami-on-hugging-face/RevAI_Scratch_Removal_Model", 
    filename="scratch_removal_test2.h5"
)
model = load_model(model_path, compile=False)
```


---

### 📁 Folder Structure

```bash
ReviveAI/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── models/
│   ├── sharpening_model.txt         # Hugging Face URL
│   └── scratch_removal_model.txt    # Hugging Face URL
│
├── notebooks/
│   ├── scratch_removal_notebook.ipynb
│   └── sharpening_model_notebook.ipynb
│
├── before_after_examples/
│   ├── sharpening/
│   └── scratch_removal/
│
├── assets/
│   └── revive banner.png, showcase images etc.

```

---

## 🧪 Training & Running the Models

ReviveAI includes end-to-end Jupyter notebooks that allow you to both **train** the models from scratch and **test** them on custom images.

### 📘 Available Notebooks

| Notebook | Description |
| -------- | ----------- |
| `sharpening_model_notebook.ipynb` | Train the sharpening (deblurring) model + Run predictions |
| `scratch_removal_notebook.ipynb` | Train the scratch removal model + Run predictions |

---

### 💡 Notebook Features

Each notebook includes:

- 🧠 **Model Architecture**  
- 🔁 **Data Loading & Preprocessing**
- 🏋️ **Training Pipeline** (with adjustable hyperparameters)
- 💾 **Saving & Exporting Weights**
- 🔍 **Evaluation**
- 🖼️ **Visual Demo on Custom Images**

---

### 🖼️ Quick Test Function (for inference)

To run a prediction on a new image (after training or loading a model), use:

```python
def display_prediction(image_path, model):
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256)) / 255.0
    input_img = np.expand_dims(img, axis=0)
    predicted = model.predict(input_img)[0]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img[..., ::-1])
    plt.title("Original Input")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(predicted)
    plt.title("Model Output")
    plt.axis("off")

    plt.show()
```

Run the function like this:

```python
display_prediction("your_image_path.jpg", model)
```

---

> ✅ Tip: If you don't want to train from scratch, you can directly load pretrained weights from Hugging Face (see [🎯 Load Pretrained Models](#-load-pretrained-models)) and skip to the testing section.

<div align="center">
  <h2>
    <b>ReviveAI</b>
  </h2>
  <h3>
    Made with ❤️ at <a href="https://github.com/ISTE-VIT" target="_blank">ISTE-VIT</a>
  </h3>
   
  ---
</div>
