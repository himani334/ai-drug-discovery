# dashboard_app.py

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem

# Load model structure
class DrugClassifier(nn.Module):
    def __init__(self):
        super(DrugClassifier, self).__init__()
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Function to convert SMILES to fingerprint
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

# Load trained model (you can retrain inside notebook and save it)
@st.cache_resource
def load_model():
    model = DrugClassifier()
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Streamlit UI
st.title("🧪 AI-Powered Drug Discovery Dashboard")
st.write("Enter a SMILES string to predict drug-target interaction:")

smiles = st.text_input("SMILES Input", "CCO")
model = load_model()

if st.button("Predict"):
    fp = smiles_to_fp(smiles)
    if fp is None:
        st.error("Invalid SMILES string. Please try again.")
    else:
        input_tensor = torch.FloatTensor(fp).view(1, -1)
        with torch.no_grad():
            prediction = model(input_tensor).item()
        st.success(f"Prediction Score: {prediction:.4f}")
        st.markdown(f"**Predicted Label:** {'Active (1)' if prediction >= 0.5 else 'Inactive (0)'}")

# Optional: Metrics Section
st.markdown("---")
st.subheader("📊 Model Metrics")
st.markdown("""
- Accuracy: **72.22%**
- AUC: **0.75**
- F1 Score: **0.77**
- Precision: **0.72**
- Recall: **0.83**
""")
