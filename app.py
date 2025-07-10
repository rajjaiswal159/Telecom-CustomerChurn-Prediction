# import necessary files
import joblib
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt


# import model
xgb_model = joblib.load('model.pkl')


# creating ui
st.header('Telecom-Customer-Churn-Prediction')

st.subheader('Enter customer age')
age = st.text_input('', key='age')
if age:
    try:
        age = int(age)
    except ValueError:
        st.error('please enter valid age')

st.subheader('Select customer Gender')
gender = st.selectbox('', ['Male', 'Female'])

st.subheader('How many months the customer has stayed with company')
tenure = st.text_input('', key='tenure')
if tenure:
    try:
        tenure = int(tenure)
    except ValueError:
        st.error('please enter valid months')

st.subheader('How many times (in a month) a customer contacted customer support')
calls = st.text_input('', key='calls')
if calls:
    try:
        calls = int(calls)
    except ValueError:
        st.error('please enter valid no. of calls')

st.subheader('How many days (in a month) that customer actively used the service')
freq = st.text_input('', key='freq')
if freq:
    try:
        freq = int(freq)
    except ValueError:
        st.error('please enter valid days')

st.subheader('Enter number of days that customer delayed their payment')
d_pay = st.text_input('', key='d_pay')
if d_pay:
    try:
        d_pay = int(d_pay)
    except ValueError:
        st.error('please enter valid days')

st.subheader('Select customer Subscription Type')
subscirpt = st.selectbox('', ['Basic', 'Premium', 'Standard'])

st.subheader('Select contract length')
contract = st.selectbox('', ['Monthly', 'Quarterly', 'Annual'])

st.subheader('How much total amount of money a customer has paid to the company')
t_spend = st.text_input('', key='t_spend')
if t_spend:
    try:
        t_spend = int(t_spend)
    except ValueError:
        st.error('please enter valid amount')

st.subheader('Enter days since last interaction with customer')
interact = st.text_input('', key='interact')
if interact:
    try:
        interact = int(interact)
    except ValueError:
        st.error('please enter valid no. of days')


# prediction 
if st.button('Prediction'):
    if not all([age, tenure, freq, calls, d_pay, t_spend, interact]):
        st.error('Please enter all numerical fields')
    else:
        st.markdown("### 📊 Model Prediction")

        # creating dataframe of inputs
        inp=pd.DataFrame({
            'Age': age,
            'Gender': gender,
            'Tenure': tenure,
            'Usage Frequency': freq,
            'Support Calls': calls,
            'Payment Delay': d_pay,
            'Subscription Type':subscirpt ,
            'Contract Length': contract,
            'Total Spend': t_spend,
            'Last Interaction': interact
        }, index=[0])

        # transform input manually
        preprocessor = xgb_model.named_steps['preprocessor']
        model = xgb_model.named_steps['model']

        x_transformed = preprocessor.transform(inp)
        features = preprocessor.get_feature_names_out()

        x_transformed_df = pd.DataFrame(x_transformed, columns=features)

        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_transformed_df)

        shap_val = shap_values[0]
        base_val = explainer.expected_value
        data_row = x_transformed_df.iloc[0]

        # prediction
        prediction = model.predict(x_transformed_df)[0]
        if prediction == 1:
            st.info('ℹ️ The model predicts: Customer will **churn**.')
        else:
            st.success('✅ The model predicts: Customer will not **churn**.')
        
        # Red and Blue SHAP Features 
        df = pd.DataFrame({
            'Feature': features,
            'SHAP Value': shap_val,
            'Value': data_row
        })
        df['Feature'] = df['Feature'].replace({
            'remainder__Tenure': 'Tenure',
            'remainder__Total Spend': 'Total Spend',
            'remainder__Usage Frequency': 'Usage Frequency',
            'remainder__Last Interaction': 'Last Interaction',
            'remainder__Age': 'Age',
            'oe__Subscription Type': 'Subscription Type',
            'remainder__Support Calls': 'Support Calls',
            'oe__Contract Length': 'Contract Length',
            'remainder__Payment Delay': 'Payment Delay',
            'ohe__Gender_Female': 'Gender'
        })


        #coverting shap values into percentage
        total_contribution = df['SHAP Value'].abs().sum()

        # Add a new column for percentage contribution
        df['% Contribution'] = (df['SHAP Value'].abs() / total_contribution * 100).round(2)

        
        # Separate red and blue features
        red_df = df[df['SHAP Value'] > 0]
        blue_df = df[df['SHAP Value'] < 0]
        
        # Show in Streamlit
        st.subheader("🔴 Features that **INCREASED** chances of churn:")
        st.dataframe(red_df[['Feature', 'Value', '% Contribution']].sort_values('% Contribution', ascending=False))
        
        st.subheader("🔵 Features that **DECREASED**  chances  of churn:")
        st.dataframe(blue_df[['Feature', 'Value', '% Contribution']].sort_values('% Contribution', ascending=False))

        st.subheader('SHAP Visualization')
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=x_transformed_df.iloc[0]), show=False)
        st.pyplot(fig)