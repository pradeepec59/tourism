import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# for model serialization
import joblib
import mlflow
import os
from google.colab import userdata
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

repo_id = "Pradeepec59/Tourism"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=userdata.get("HF_TOKEN"))

df = pd.read_csv("tourism_project/data/tourism.csv")
print("Dataset loaded successfully")

# Drop unique identifier column (not useful for modeling)
df.drop(columns=['CustomerID'], inplace=True)

# Encode categorical columns
categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
label_encoder_instance = LabelEncoder() # Renamed to avoid conflict
for col in categorical_cols:
    df[col] = label_encoder_instance.fit_transform(df[col])
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
    X, y, test_size=0.2, random_state=42)

# Define numeric and categorical features
numeric_features = [
    'Age', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 'NumberOfFollowups', 'DurationOfPitch', 'PitchSatisfactionScore',
    'MonthlyIncome', 'NumberOfTrips', 'NumberOfChildrenVisiting'
]

categorical_features = categorical_cols

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost Regressor
xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

# Hyperparameter grid
param_grid = {
    'xgbregressor__n_estimators': [50, 100],
    'xgbregressor__max_depth': [3, 5],
    'xgbregressor__learning_rate': [0.01, 0.05],
    'xgbregressor__subsample': [0.7, 0.8],
    'xgbregressor__colsample_bytree': [0.7, 0.8],
    'xgbregressor__reg_lambda': [0.1, 1]
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Grid Search
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(Xtrain, ytrain)

    # Log parameter sets
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_neg_mse", mean_score)

    # Best model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Metrics
    train_rmse = mean_squared_error(ytrain, y_pred_train)
    test_rmse = mean_squared_error(ytest, y_pred_test)

    train_mae = mean_absolute_error(ytrain, y_pred_train)
    test_mae = mean_absolute_error(ytest, y_pred_test)

    train_r2 = r2_score(ytrain, y_pred_train)
    test_r2 = r2_score(ytest, y_pred_test)

    # Log metrics
    mlflow.log_metrics({
        "train_RMSE": train_rmse,
        "test_RMSE": test_rmse,
        "train_MAE": train_mae,
        "test_MAE": test_mae,
        "train_R2": train_r2,
        "test_R2": test_r2
    })

    # Save the model locally
model_path = "tourism_project/tourism_project_test.joblib"
joblib.dump(best_model, model_path)

# Log the model artifact
mlflow.log_artifact(model_path, artifact_path="model")
print(f"Model saved as artifact at: {model_path}")

# Upload to Hugging Face
model_repo_id = "Pradeepec59/Tourism" # Assuming the model is uploaded to the same repo, but specified as 'model' type
model_repo_type = "model"

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=model_repo_id, repo_type=model_repo_type)
    print(f"Space '{model_repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{model_repo_id}' not found. Creating new space...")
    create_repo(repo_id=model_repo_id, repo_type=model_repo_type, private=False)
    print(f"Space '{model_repo_id}' created.")

api.upload_file(
    path_or_fileobj="tourism_project/tourism_project_test.joblib", # Corrected path
    path_in_repo="tourism_project_test.joblib",
    repo_id=model_repo_id,
    repo_type=model_repo_type,
)
