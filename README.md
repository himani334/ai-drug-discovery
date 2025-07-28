
# 🧪 AI-Powered Drug Discovery Using Deep Learning

This project demonstrates a complete deep learning pipeline for predicting drug-target interactions using SMILES molecular representations and PyTorch neural networks.

## 🚀 Features
- SMILES to Morgan fingerprint conversion using RDKit
- Custom PyTorch binary classification model
- Stratified K-Fold cross-validation
- Evaluation metrics: Accuracy, AUC, F1 Score, Precision, Recall
- Streamlit dashboard for real-time predictions

## 📁 Project Structure

```
AI_Drug_Discovery_Project/
├── dashboard_app.py               # Streamlit app
├── Drug_Discovery_DeepLearning_Himani_Raval.ipynb  # Jupyter notebook
├── model.pth                      # Trained PyTorch model
├── drug_target_sample_dataset.csv # Example dataset
├── Draft_Paper_AI_Drug_Discovery_Himani_Raval.docx # Methodology and results
├── Research_Report_AI_Drug_Discovery_Himani_Raval.docx # Literature review
├── DIAGRAM.png                    # Model architecture diagram
└── README.md                      # Project overview (this file)
```

## 💻 Run the Dashboard

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the Streamlit app:
```bash
streamlit run dashboard_app.py
```

## 🧬 Model Architecture

Feedforward Neural Network (2048 → 512 → 128 → 1) trained on fingerprint vectors generated from SMILES.

## 📝 Example Input

```
SMILES: CCO
Prediction: Active (1)
```

## 📊 Results

- Accuracy: **72.22%**
- AUC: **0.75**
- F1 Score: **0.77**
- Precision: **0.72**
- Recall: **0.83**

## 📜 License

Apache 2.0 — for academic and research use.
