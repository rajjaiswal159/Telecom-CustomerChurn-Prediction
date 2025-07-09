# 📉 Telecom Customer Churn Prediction

An end-to-end machine learning project to predict customer churn in the telecom industry. This project helps identify users likely to cancel their subscription so that retention strategies can be applied proactively.


## 🚀 Project Objective

To build a robust and interpretable churn prediction model using supervised learning algorithms that can:

- Detect customers likely to churn.
- Help reduce customer loss.
- Provide actionable business insights.


## 🧾 Dataset Overview

- **Records:** 500,000+
- **Features:** 11 (including categorical and numerical features)
- **Target Variable:** `Churn` (Yes/No)


## 🔧 Technologies Used

- Python
- pandas, NumPy
- Scikit-learn
- XGBoost, LightGBM, Random Forest
- Optuna (for hyperparameter tuning)
- Matplotlib, Seaborn (for visualization)
- Google Colab (for development)


## 📊 Workflow

### 1. **Exploratory Data Analysis (EDA)**
- Univariate and bivariate analysis
- Churn rate distribution
- Correlation between features

### 2. **Data Preprocessing**
- Handling missing values
- Encoding categorical features using:
  - `OneHotEncoder` for nominal
  - `OrdinalEncoder` for ordinal
- Created pipelines using `ColumnTransformer`

### 3. **Model Building**
- Baseline model
  - XGBClassifier
  - RandomForestClassifier
  - LGBMClassifier
- Performance metrics:
  - Classification Report
  - Confusion Matrix
  - F1 Score

### 4. **Hyperparameter Tuning**
- Used **Optuna** for:
  - Efficient and scalable tuning
  - Parallel trials
- Trained final models with best hyperparameters

### 5. **Feature Importance**
- Visualized top features contributing to churn
- Explained model interpretability using built-in `.feature_importances_`


## 🏆 Results

- **Best Model:** XGBClassifier (with Optuna tuning)
- **F1 Score:** ~95%
- **Key Features:** Contract length, subscription type, gender, etc.


## 📁 Folder Structure

📂 Telecom-CustomerChurn-Prediction/ ├── CustomerChurn.ipynb       # Main notebook ├── requirements.txt          # Required libraries ├── README.md                 # Project documentation ├── data/                     # (Optional) Dataset if permitted └── visuals/                  # Confusion matrix, feature importance, etc.

---

## 📌 Key Learnings

- How to structure a real-world machine learning workflow
- Efficient hyperparameter tuning using Optuna
- Importance of preprocessing pipelines for production-readiness
- Model comparison and selection using evaluation metrics

---

## 📎 To Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Telecom-CustomerChurn-Prediction.git

2. Install dependencies:

pip install -r requirements.txt


3. Open and run the notebook:

jupyter notebook CustomerChurn.ipynb




---

📬 Connect With Me

📧 Email: Rajjeswal30@gmail.com@example.com

🔗 LinkedIn: https://www.linkedin.com/in/raj-jaiswal-45604330a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app




⭐ If you like this project, feel free to star it!
