# Fakenews_prediction
A deep learning project using BERT
# 📰 Fake News Detection using BERT
This project implements a **Fake News Detection** model using **BERT (Bidirectional Encoder Representations from Transformers)** to classify news articles as *Real* or *Fake*.
It includes model training, evaluation, prediction, and a web-based **Streamlit UI** for interactive testing.

# Features
* BERT-based deep learning model for text classification
* End-to-end training and inference pipeline
* Interactive Streamlit web app for prediction
* Clean and modular code structure
* CUDA support for GPU acceleration

## 📁 Project Structure

```
D:\IITJ\Project_dl
│
├── data/                     # Folder for dataset (ignored in .gitignore)
├── model.py                  # Model architecture definition (BERT Classifier)
├── data_utils.py             # Preprocessing and dataset utilities
├── train.py                  # Training logic
├── evaluate.py               # Model evaluation code
├── predict.py                # CLI-based prediction script
├── app.py                    # Streamlit web interface
├── main.py                   # Main training entry point
├── prepare_dataset.ipynb     # Notebook for dataset creation
├── dataset.ipynb             # Notebook for data exploration
├── requirements.txt          # Dependencies list
└── README.md                 # Project documentation
```

---

## Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/Akkalesh/Fakenews_prediction.git
cd Fakenews_prediction
```

### Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training the Model

Run:

```bash
python main.py
```

This will:

* Load and preprocess data
* Fine-tune BERT for fake news classification
* Save the trained model as `bert_fake_news_model.pth`

## Prediction (Command Line)

Use the trained model for quick predictions:

```bash
python predict.py
```

## Streamlit Web App

Run the interactive app:

```bash
streamlit run app.py
```

Then open the local URL (usually [http://localhost:8501/](http://localhost:8501/)) in your browser.

Type or paste any news headline or article to check if it’s **Fake** or **Real**.

---

## 📦 Dependencies

Main libraries:

```
torch
transformers
pandas
scikit-learn
streamlit
tqdm
numpy
```

Install everything using:

```bash
pip install -r requirements.txt
```

---

## Results

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 99.9% |
| Precision | 1.00  |
| Recall    | 1.00  |
| F1-Score  | 1.00  |

