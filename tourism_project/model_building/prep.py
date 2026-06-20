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

from google.colab import userdata

repo_id = "Pradeepec59/Tourism"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=userdata.get("HF_TOKEN"))

DATASET_PATH = "hf://datasets/Pradeepec59/Tourism/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unique identifier column (not useful for modeling)
df.drop(columns=['CustomerID'], inplace=True)

# Encode categorical columns
categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
OneHotEncoder = LabelEncoder()
for col in categorical_cols:
    df[col] = OneHotEncoder.fit_transform(df[col])
print("Categorical columns encoded successfully.")

#replace 'Fe male' as 'Female' in Gender column
df['Gender'] = df['Gender'].replace('Fe male', 'Female')
print("Gender column updated successfully.")

#replace 'Single' as 'Unmarried' in MaritalStatus column
df['MaritalStatus'] = df['MaritalStatus'].replace('Single', 'Unmarried')
print("MaritalStatus column updated successfully.")

# Define target variable
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data split into training and testing sets.")

# Create the data folder if it doesn't exist
os.makedirs("tourism_project/data", exist_ok=True)

Xtrain.to_csv("tourism_project/data/Xtrain.csv",index=False)
Xtest.to_csv("tourism_project/data/Xtest.csv",index=False)
ytrain.to_csv("tourism_project/data/ytrain.csv",index=False)
ytest.to_csv("tourism_project/data/ytest.csv",index=False)
print("Train and test datasets saved locally.")

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=f"tourism_project/data/{file_path}",
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Pradeepec59/Tourism",
        repo_type="dataset",
    )
print("Train and test datasets uploaded to Hugging Face Hub.")
