import joblib
import pandas as pd
import streamlit as st

# Load the model
model = joblib.load(open('loan_prediction_model.pkl', 'rb'))


# Define the necessary features for bank loan prediction
necessary_features = ['Income', 'CreditCard', 'Education', 'CD Account', 'Mortgage']

def main():
    # Add a title
    st.title("Bank Loan Prediction")

    # Define the input fields
    income = st.number_input("Income (in thousands)", min_value=0)
    credit_card = st.selectbox("Credit Card", ["Yes", "No"])
    education = st.selectbox("Education Level", [1, 2, 3])
    cd_account = st.selectbox("CD Account", ["Yes", "No"])
    mortgage = st.number_input("Mortgage (in thousands)", min_value=0)

    # Map string values to numeric representations
    credit_card_mapping = {"Yes": 1, "No": 0}
    cd_account_mapping = {"Yes": 1, "No": 0}

    # Create a dictionary with user inputs
    input_data = {
        "Income": income,
        "CreditCard": credit_card_mapping[credit_card],
        "Education": education,
        "CD Account": cd_account_mapping[cd_account],
        "Mortgage": mortgage
    }

    # Convert the input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Select only the necessary features
    input_df = input_df[necessary_features]

    # Make predictions using the loaded model
    prediction = model.predict(input_df)

    # Display the prediction result
    if prediction[0]:
        st.write("Congratulations! The customer is likely to accept the personal loan.")
    else:
        st.write("The customer is unlikely to accept the personal loan.")

# Run the Streamlit app
if _name_ == '_main_':
    main()
