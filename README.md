# Emotion Detection using Transformers

This project applies **Deep Learning with Transformer models** to detect multiple emotions from text data. The fine-tuned model participated in the **Kaggle '2025 Sep DL & GenAI Project'**, where it achieved a top score of **0.838** (Macro F1-Score).

---

## 🧠 Problem Overview
The dataset contains short English sentences labeled with **five emotion categories**:
- 😠 Anger  
- 😨 Fear  
- 😀 Joy  
- 😢 Sadness  
- 😮 Surprise  

Each text can have **multiple emotions simultaneously** (multi-label classification).  
The task was to build models that predict these emotions accurately, with **Macro F1-Score** as the evaluation metric.

---

## 📊 Exploratory Data Analysis (EDA)
Key takeaways from EDA:
- **~16 words per text** on average — short and conversational in nature.
- Emotion distribution was **imbalanced**:  
  - Most common: `fear` (3860 examples)  
  - Least common: `anger` (~800 examples)
- **Correlations:**  
  - Strong negative correlation between `joy` and `fear` (**-0.47**).
  - Moderate positive correlation between `fear` and `sadness` (**0.29**).

The dataset was relatively clean; no major noise or missing values.

---

## 🧹 Preprocessing
- Converted text to lowercase.
- Removed URLs, HTML tags, and social handles.
- Tokenized using model-specific tokenizers (BERT and RoBERTa).
- No heavy text cleaning to preserve emotional tone.

---

## 🛠️ Modeling Approach

### 1. Initial Baseline — `bert-base-uncased`
- Fine-tuned with:
  - LR: \(3 \times 10^{-5}\)  
  - Epochs: 3  
  - Batch size: 16  
- Scored: **0.772**

### 2. Improved Model — `roberta-base`
- Fine-tuned with:
  - LR: \(1 \times 10^{-5}\)  
  - Epochs: 5  
  - Batch size: 16  
  - Linear LR Warmup + Scheduler  
- Scored: **0.81**

### 3. Final Tuning
- Increased training to **15 epochs**, with same LR (1e-5)  
- Kept warm-up + scheduler  
- Fixed random seeds for reproducibility  
- Final score: **0.863** ✅

---

## 🏁 Final Results
| Model | LR | Epochs | Score (Macro F1) |
|--------|------|--------|------------------|
| BERT (baseline) | 3e-5 | 3 | 0.772 |
| RoBERTa (tuned) | 1e-5 | 5 | 0.810 |
| RoBERTa (optimized) | 1e-5 | 10 | **0.849** |
| RoBERTa (optimized) | 1e-5 | 15 | **0.858** |
| RoBERTa (optimized) | 2e-5 | 12 | **0.863** |

---

## 💻 Tools & Libraries
- PyTorch
- Hugging Face Transformers
- NumPy, Pandas, Matplotlib
- scikit-learn

---

## ✨ Next Steps
- Try ensemble of DeBERTa + RoBERTa
- Experiment with class-weighted loss or focal loss for imbalance
- Add back-translation based data augmentation

---

## 🧠 Author
Developed by Abhishek Guru as part of the Kaggle 2025 Sep DL & GenAI Competition.

---

