# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# ✅ Works in both Colab and GitHub Actions
try:
    from google.colab import userdata
    hf_token = userdata.get("HF_TOKEN")
except ImportError:
    hf_token = os.environ.get("HF_TOKEN")

if hf_token is None:
    raise ValueError("HF_TOKEN not found in Colab Secrets or GitHub Actions Secrets.")

repo_id = "Pradeepec59/Tourism"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=hf_token)

# ✅ Pass token explicitly for hf:// path access
from huggingface_hub import login
login(token=hf_token)

DATASET_PATH = "hf://datasets/Pradeepec59/Tourism/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unique identifier column
df.drop(columns=['CustomerID'], inplace=True)

# ✅ Fix: Apply replacements BEFORE encoding
df['Gender'] = df['Gender'].replace('Fe male', 'Female')
print("Gender column updated successfully.")
df['MaritalStatus'] = df['MaritalStatus'].replace('Single', 'Unmarried')
print("MaritalStatus column updated successfully.")

# Encode categorical columns
categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])
print("Categorical columns encoded successfully.")

# Define target variable
target_col = 'ProdTaken'

# Split into X and y
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data split into training and testing sets.")

# Save locally
os.makedirs("tourism_project/data", exist_ok=True)
Xtrain.to_csv("tourism_project/data/Xtrain.csv", index=False)
Xtest.to_csv("tourism_project/data/Xtest.csv", index=False)
ytrain.to_csv("tourism_project/data/ytrain.csv", index=False)
ytest.to_csv("tourism_project/data/ytest.csv", index=False)
print("Train and test datasets saved locally.")

# Upload to HF
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for file_path in files:
    api.upload_file(
        path_or_fileobj=f"tourism_project/data/{file_path}",
        path_in_repo=file_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
print("Train and test datasets uploaded to Hugging Face Hub.")
