# 📉 Customer Churn Prediction with Explainable AI (XAI)

Predicting customer churn helps businesses retain valuable users and reduce revenue loss. This project leverages machine learning to predict customer churn and integrates **Explainable AI (XAI)** techniques to interpret the model's decisions.

---

## 🚀 Overview

In this project, we:
- Perform end-to-end **data preprocessing**.
- Train multiple **ML models** for churn prediction.
- Use **SHAP** (SHapley Additive exPlanations) and **LIME** (Local Interpretable Model-agnostic Explanations) for model explainability.
- Visualize key insights to drive business decisions.

---

## 📁 Project Structure

customer-churn-xai/
├── data/
│ └── churn_data.csv
├── notebooks/
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_model_training.ipynb
│ ├── 03_xai_analysis.ipynb
├── src/
│ ├── utils.py
│ ├── model.py
│ └── explainability.py
├── results/
│ └── shap_summary_plot.png
├── README.md
└── requirements.txt


---

## 🧠 Machine Learning Algorithms Used

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

The best-performing model is selected based on evaluation metrics.

---

## 📊 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix

> ✅ Achieved an accuracy of **~93%** and using the XGBoost classifier.

---

## 💡 Explainable AI (XAI)

To ensure **trust and transparency**, we used:

### ✅ SHAP
- Global feature importance
- Individual prediction explanations

### ✅ LIME
- Local interpretations
- Explains what influenced a particular prediction

---

## 📌 Key Insights

- Tenure, Monthly Charges, and Contract Type were the most influential features.
- Customers with **month-to-month contracts**, **high charges**, and **low tenure** were more likely to churn.

---

## 🛠️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/customer-churn-xai.git
cd customer-churn-xai
pip install -r requirements.txt
▶️ How to Run
Launch Jupyter Notebook:


jupyter notebook
Follow the notebooks in order:

01_data_preprocessing.ipynb

02_model_training.ipynb

03_xai_analysis.ipynb

🧪 Requirements
Python 3.8+

pandas, numpy, scikit-learn

matplotlib, seaborn

xgboost, lightgbm

shap, lime

Install via:

pip install -r requirements.txt
📎 Dataset
The dataset used is publicly available from the Telco Customer Churn Dataset on Kaggle.

📸 Sample SHAP Plot

📢 Conclusion
This project demonstrates not only how to predict churn using robust ML techniques but also how to interpret and explain predictions, empowering data-driven and transparent business decisions.

🤝 Let's Connect
Made with ❤️ by Raj Jaiswal

📄 License
This project is licensed under the MIT License.


---

Let me know if you want a **version with Streamlit or FastAPI**, or if you'd like help **customizing it to your actual GitHub repo or project**.
