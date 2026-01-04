# Quora Question Pair Similarity 🤖

An NLP-based machine learning project that predicts whether two questions asked on Quora have the same intent (duplicates) or are different.

## 🚀 Live Demo
You can run this project locally to test question pairs. The system uses a **Random Forest Classifier** combined with **Fuzzy String Matching** to detect semantic similarity even when wordings are slightly different.

## ✨ Features
- **Text Preprocessing:** Automated cleaning of special characters and text normalization.
- **Feature Engineering:** Extracted 8 key features including:
  - Basic Features: Character Length, Word Count.
  - Advanced Features: FuzzyWuzzy scores (QRatio, Partial Ratio, Token Set Ratio, Token Sort Ratio).
- **Machine Learning:** Random Forest model trained on a balanced subset of 20,000 question pairs.
- **Web Interface:** Interactive UI built with **Streamlit**.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, Scikit-learn, FuzzyWuzzy, NumPy
- **Deployment:** Streamlit
- **Environment:** VS Code / Jupyter Notebooks

## 📊 Performance
- **Model:** Random Forest Classifier
- **Accuracy:** ~72% (on a balanced 20k dataset)

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mithilesh-cloud/nlp-semantic-quora.git