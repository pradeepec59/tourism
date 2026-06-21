import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# ✅ Works in both Colab and GitHub Actions and HF Space
try:
    from google.colab import userdata
    hf_token = userdata.get("HF_TOKEN")
except ImportError:
    hf_token = os.environ.get("HF_TOKEN")

# ✅ On HF Space, token is optional for public repos
api = HfApi(token=hf_token)

# Download and load the trained model
model_path = hf_hub_download(
    repo_id="Pradeepec59/Tourism-App",
    filename="tourism_project_test.joblib",
    repo_type="model",
    token=hf_token
)
model = joblib.load(model_path)

# Download original data for LabelEncoder fitting
original_data_path = hf_hub_download(
    repo_id="Pradeepec59/Tourism",
    filename="tourism.csv",
    repo_type="dataset",
    token=hf_token
)
original_df = pd.read_csv(original_data_path)

# Apply same replacements as prep.py
original_df['Gender'] = original_df['Gender'].replace('Fe male', 'Female')
original_df['MaritalStatus'] = original_df['MaritalStatus'].replace('Single', 'Unmarried')

# Streamlit UI
st.title("Tourism Package Prediction")
st.write("This application predicts the likelihood of a customer purchasing a tourism package.")

st.header("User Input Features")

categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']

# Fit LabelEncoders
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(original_df[col].astype(str).unique())
    label_encoders[col] = le

# Numerical inputs
input_age               = st.number_input("Age", min_value=18, max_value=90, value=30)
input_num_persons       = st.number_input("Number of Persons Visiting", min_value=1, max_value=8, value=2)
input_property_star     = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
input_num_followups     = st.number_input("Number of Followups", min_value=0, max_value=10, value=2)
input_duration_pitch    = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=15)
input_pitch_satisfaction= st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
input_monthly_income    = st.number_input("Monthly Income", min_value=0.0, max_value=100000.0, value=5000.0)
input_num_trips         = st.number_input("Number of Trips", min_value=1, max_value=20, value=5)
input_num_children      = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
input_city_tier         = st.number_input("City Tier", min_value=1, max_value=3, value=1)
input_passport          = st.selectbox("Passport", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
input_own_car           = st.selectbox("Own Car", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Categorical inputs
input_type_of_contact = st.selectbox("Type of Contact", options=label_encoders['TypeofContact'].classes_)
input_occupation      = st.selectbox("Occupation", options=label_encoders['Occupation'].classes_)
input_gender          = st.selectbox("Gender", options=label_encoders['Gender'].classes_)
input_product_pitched = st.selectbox("Product Pitched", options=label_encoders['ProductPitched'].classes_)
input_marital_status  = st.selectbox("Marital Status", options=label_encoders['MaritalStatus'].classes_)
input_designation     = st.selectbox("Designation", options=label_encoders['Designation'].classes_)

# Assemble input DataFrame
expected_columns = [
    'Unnamed: 0', 'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch',
    'Gender', 'MaritalStatus', 'MonthlyIncome', 'NumberOfChildrenVisiting',
    'NumberOfFollowups', 'NumberOfPersonVisiting', 'NumberOfTrips',
    'Occupation', 'Passport', 'PitchSatisfactionScore', 'PreferredPropertyStar',
    'ProductPitched', 'OwnCar', 'Designation'
]

input_data_dict = {
    'Unnamed: 0': 0,
    'Age': input_age,
    'TypeofContact': label_encoders['TypeofContact'].transform([input_type_of_contact])[0],
    'CityTier': input_city_tier,
    'DurationOfPitch': input_duration_pitch,
    'Gender': label_encoders['Gender'].transform([input_gender])[0],
    'MaritalStatus': label_encoders['MaritalStatus'].transform([input_marital_status])[0],
    'MonthlyIncome': input_monthly_income,
    'NumberOfChildrenVisiting': input_num_children,
    'NumberOfFollowups': input_num_followups,
    'NumberOfPersonVisiting': input_num_persons,
    'NumberOfTrips': input_num_trips,
    'Occupation': label_encoders['Occupation'].transform([input_occupation])[0],
    'Passport': input_passport,
    'PitchSatisfactionScore': input_pitch_satisfaction,
    'PreferredPropertyStar': input_property_star,
    'ProductPitched': label_encoders['ProductPitched'].transform([input_product_pitched])[0],
    'OwnCar': input_own_car,
    'Designation': label_encoders['Designation'].transform([input_designation])[0]
}

input_data = pd.DataFrame([input_data_dict])[expected_columns]

# Predict
if st.button("Predict Tourism Package"):
    prediction = model.predict(input_data)[0]
    purchase_likelihood = prediction * 100
    st.subheader("Prediction Result:")
    st.success(f"Predicted likelihood of purchasing the package: **{purchase_likelihood:.2f}%**")
