📉 Telecom Customer Churn Prediction
An end-to-end machine learning project to predict customer churn in the telecom industry. This project helps identify users likely to cancel their subscription so that retention strategies can be applied proactively.

🚀 Project Objective
To build a robust and interpretable churn prediction model using supervised learning algorithms that can:

Detect customers likely to churn

Help reduce customer loss

Provide actionable business insights

🧾 Dataset Overview
Records: 500,000+

Features: 11 (including categorical and numerical features)

Target Variable: Churn (Yes/No)

🔧 Technologies Used
Python

Libraries: pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Random Forest, Optuna

Visualization: Matplotlib, Seaborn

Platform: Google Colab (for development), Streamlit (for deployment)

📊 Workflow
1. Exploratory Data Analysis (EDA)
Univariate and bivariate analysis

Churn rate distribution

Correlation between features

2. Data Preprocessing
Handled missing values

Encoded categorical features using:

OneHotEncoder for nominal features

OrdinalEncoder for ordinal features

Built pipelines with ColumnTransformer

3. Model Building
Tried baseline models:

XGBClassifier, RandomForestClassifier, LGBMClassifier

Evaluated using:

Classification Report

Confusion Matrix

F1 Score

4. Hyperparameter Tuning
Used Optuna for efficient tuning:

Pruning

Parallel trials

5. Feature Importance & Interpretability
Used .feature_importances_ for model explanation

Used SHAP values for local explanations

🏆 Results
Best Model: XGBClassifier (with Optuna tuning)

F1 Score: ~95%

Key Features: Contract Length, Subscription Type, Gender, Tenure, etc.

📁 Folder Structure
bash
Copy
Edit
Telecom-CustomerChurn-Prediction/
├── CustomerChurn.ipynb         # Main notebook
├── requirements.txt            # Required libraries
├── app.py                      # Streamlit app code
├── model.pkl                   # Trained pipeline (preprocessor + model)
├── README.md                   # Project documentation
├── data/                       # (Optional) Dataset
└── visuals/                    # SHAP plots, confusion matrix, etc.
🌐 Streamlit Web App
An interactive Streamlit UI was built to allow real-time churn predictions and SHAP-based explanations.

✅ Features:
User inputs customer data

Predicts churn using a trained XGBoost model

Explains each prediction using SHAP waterfall plot

Displays top features increasing/decreasing churn probability

🛠 How to Run the App Locally
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/your-username/Telecom-CustomerChurn-Prediction.git
cd Telecom-CustomerChurn-Prediction
2. Install the dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Ensure the model file (model.pkl) is present
This file should contain a trained pipeline with:

'preprocessor': a preprocessing step (e.g., ColumnTransformer)

'model': the trained XGBoost model

4. Run the Streamlit app
bash
Copy
Edit
streamlit run app.py
📦 Example requirements.txt
nginx
Copy
Edit
streamlit
pandas
shap
matplotlib
joblib
scikit-learn
xgboost
📌 Key Learnings
Real-world ML pipeline building and deployment

SHAP for model transparency and explanation

Using Optuna for scalable hyperparameter tuning

Streamlit deployment with proper preprocessing integration

📎 To Run the Jupyter Notebook
Open the notebook:

bash
Copy
Edit
jupyter notebook CustomerChurn.ipynb
📬 Connect With Me
📧 Email: Rajjeswal30@gmail.com@example.com
🔗 LinkedIn: Raj Jaiswal

⭐ If you like this project, feel free to star it!
