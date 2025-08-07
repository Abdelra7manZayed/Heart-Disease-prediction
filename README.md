# 🧠 Comprehensive Machine Learning Pipeline for Heart Disease Prediction

## 📌 Project Overview

This project implements a complete machine learning workflow on the **Heart Disease UCI Dataset** to predict and analyze heart disease risks. It combines both **supervised** and **unsupervised** learning models, detailed data preprocessing, **dimensionality reduction**, **feature selection**, and a **Streamlit-based UI** for real-time predictions. The entire pipeline is designed for deployment, reproducibility, and ease of use.

---

## 🎯 Objectives

- Clean and preprocess the dataset (handle missing values, encoding, scaling).
- Reduce dimensionality using **Principal Component Analysis (PCA)**.
- Select significant features using statistical and machine learning-based methods.
- Train and evaluate multiple classification models:
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - Support Vector Machines (SVM)
- Discover hidden patterns with:
  - **K-Means Clustering**
  - **Hierarchical Clustering**
- Optimize model performance with **GridSearchCV** and **RandomizedSearchCV**.
- Develop a user-friendly **Streamlit web UI** for real-time predictions (Bonus).
- Host the app using **Ngrok** (Bonus).
- Upload and document the project on **GitHub** for public access.

---

## 🧰 Tools & Technologies

- **Languages:** Python
- **Libraries:**
  - Data Manipulation: `Pandas`, `NumPy`
  - Visualization: `Matplotlib`, `Seaborn`
  - Modeling: `Scikit-learn`, `XGBoost` (optional), `TensorFlow/Keras` (optional)
  - UI/Deployment: `Streamlit`, `Ngrok`, `joblib`, `pickle`
- **Techniques:**
  - Feature Selection: RFE, Chi-Square Test, Feature Importance
  - Dimensionality Reduction: PCA
  - Hyperparameter Tuning: GridSearchCV, RandomizedSearchCV

---

## 🧪 Project Workflow

### 1. 📊 Data Preprocessing

- Load dataset into a Pandas DataFrame
- Handle missing values and encode categorical data
- Standardize features with MinMaxScaler or StandardScaler
- Perform EDA (histograms, correlation heatmaps, boxplots)

**✅ Deliverable:** Cleaned dataset ready for modeling

---

### 2. 🔻 Dimensionality Reduction (PCA)

- Apply PCA to reduce feature space while retaining variance
- Determine optimal components using explained variance ratio
- Visualize results with scatter plots and variance plots

**✅ Deliverable:** PCA-transformed dataset and visualizations

---

### 3. ⭐ Feature Selection

- Rank features using:
  - Feature Importance (Random Forest/XGBoost)
  - Recursive Feature Elimination (RFE)
  - Chi-Square Test
- Select top features for modeling

**✅ Deliverable:** Dataset with reduced features and feature importance visualizations

---

### 4. 🤖 Supervised Learning (Classification)

- Split dataset (80% train / 20% test)
- Train models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - SVM
- Evaluate with:
  - Accuracy, Precision, Recall, F1-score
  - ROC Curve and AUC

**✅ Deliverable:** Trained models with performance metrics

---

### 5. 🔍 Unsupervised Learning (Clustering)

- Apply K-Means (elbow method for K)
- Perform Hierarchical Clustering (dendrograms)
- Compare cluster results with actual labels

**✅ Deliverable:** Clustering results and visualizations

---

### 6. 🧪 Hyperparameter Tuning

- Optimize models using:
  - GridSearchCV
  - RandomizedSearchCV
- Compare tuned models vs. baseline

**✅ Deliverable:** Optimized model with best hyperparameters

---

### 7. 💾 Model Export & Deployment

- Save model as `.pkl` using `joblib` or `pickle`
- Ensure full pipeline export (preprocessing + model)

**✅ Deliverable:** Exported trained model (.pkl)

---

### 8. 🌐 Streamlit Web App [Bonus]

- Build Streamlit UI for user input and predictions
- Add visualizations for user insight

**✅ Deliverable:** Functional and interactive UI

---

### 9. 🚀 Ngrok Deployment [Bonus]

- Deploy Streamlit app locally
- Expose via Ngrok for public access

**✅ Deliverable:** Ngrok public URL for app access

---


## 📁 Project Structure

```
Heart_Disease_Project/
│
├── data/
│   └── heart_disease.csv
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
│
├── models/
│   └── final_model.pkl
│
├── ui/
│   └── app.py
│
├── deployment/
│   └── ngrok_setup.txt
│
├── results/
│   └── evaluation_metrics.txt
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

- **Name:** Heart Disease UCI Dataset  
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

---

## ✅ Final Deliverables

- ✅ Cleaned and preprocessed dataset
- ✅ PCA-transformed data
- ✅ Feature-selected dataset
- ✅ Trained supervised and unsupervised models
- ✅ Evaluation metrics and performance reports
- ✅ Hyperparameter-tuned model
- ✅ Exported `.pkl` model file
- ✅ Functional Streamlit Web App [Bonus]
- ✅ Ngrok public deployment link [Bonus]

