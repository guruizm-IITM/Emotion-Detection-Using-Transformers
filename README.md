# Emotion Detection with RoBERTa

This project applies Deep Learning and Transformer models to detect multiple emotions from text. Using RoBERTa fine-tuned for multi-label emotion classification, the model achieved a leaderboard score of **0.81** on the Kaggle “2025 Sep DL & GenAI Project” competition.

## 🧠 Model Overview
- **Architecture:** RoBERTa-base  
- **Learning Rate:** 1e-5  
- **Epochs:** 5  
- **Task Type:** Multi-label Emotion Classification  
- **Framework:** PyTorch & Hugging Face Transformers  

## 📊 Dataset
The dataset contains text samples labeled with multiple emotions — *anger, fear, joy, sadness, surprise*.  
Preprocessing involved tokenization and padding using RoBERTa’s tokenizer.

## 🚀 Training
Training used GPU acceleration for faster convergence and BCEWithLogits loss for handling multi-label outputs.

## 🧾 Results
| Metric | Score |
|:-------:|:------:|
| Leaderboard | **0.81** |

## 📂 Files
- `train.csv` – Training data  
- `test.csv` – Test data  
- `submission.csv` – Final predictions  

## 🧠 Author
Developed by Abhishek Guru as part of the Kaggle 2025 Sep DL & GenAI Competition.

---

