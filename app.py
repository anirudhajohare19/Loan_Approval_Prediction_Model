import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('loan_model.pkl')

# App title
st.title(" Loan Approval Prediction App")

# Input form
st.subheader("Enter Applicant Details:")

person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Annual Income", value=50000)
person_home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
person_emp_length = st.slider("Employment Length (Years)", 0, 20, 5)
loan_intent = st.selectbox("Loan Intent", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])
loan_grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
loan_amnt = st.number_input("Loan Amount", value=10000)
loan_int_rate = st.number_input("Interest Rate (%)", value=10.5)
loan_percent_income = st.number_input("Loan % of Income", value=0.2)
cb_person_default_on_file = st.selectbox("Previously Defaulted", ['Y', 'N'])
cb_person_cred_hist_length = st.slider("Credit History Length", 1, 30, 5)

# Encode categorical features manually (should match label encoding used in training)
home_ownership_map = {'RENT': 3, 'OWN': 1, 'MORTGAGE': 2, 'OTHER': 0}
loan_intent_map = {'EDUCATION': 0, 'MEDICAL': 1, 'VENTURE': 5, 'PERSONAL': 3, 'DEBTCONSOLIDATION': 2, 'HOMEIMPROVEMENT': 4}
loan_grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
default_map = {'Y': 1, 'N': 0}

# Prepare input
features = np.array([[
    person_age,
    person_income,
    home_ownership_map[person_home_ownership],
    person_emp_length,
    loan_intent_map[loan_intent],
    loan_grade_map[loan_grade],
    loan_amnt,
    loan_int_rate,
    loan_percent_income,
    default_map[cb_person_default_on_file],
    cb_person_cred_hist_length
]])

# Prediction
if st.button("Predict Loan Status"):
    result = model.predict(features)[0]
    if result == 1:
        st.success(" Loan Approved")
    else:
        st.error(" Loan Rejected")