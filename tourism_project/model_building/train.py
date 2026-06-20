import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import mlflow
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ✅ Works in both Colab and GitHub Actions
try:
    from google.colab import userdata
    hf_token = userdata.get("HF_TOKEN")
except ImportError:
    hf_token = os.environ.get("HF_TOKEN")

if hf_token is None:
    raise ValueError("HF_TOKEN not found in Colab Secrets or GitHub Actions Secrets.")

# Initialize API client
api = HfApi(token=hf_token)
login(token=hf_token)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism")

# ✅ Load from HF dataset (not local path — train.py runs on fresh runner)
from huggingface_hub import hf_hub_download

def download_csv(filename):
    path = hf_hub_download(
        repo_id="Pradeepec59/Tourism",
        filename=filename,
        token=hf_token,
        repo_type="dataset"
    )
    return pd.read_csv(path)

Xtrain = download_csv("Xtrain.csv")
Xtest  = download_csv("Xtest.csv")
ytrain = download_csv("ytrain.csv")
ytest  = download_csv("ytest.csv")
print("Datasets loaded successfully from Hugging Face.")

# Define numeric and categorical features
numeric_features = [
    'Age', 'NumberOfPersonVisiting', 'PreferredPropertyStar',
    'NumberOfFollowups', 'DurationOfPitch', 'PitchSatisfactionScore',
    'MonthlyIncome', 'NumberOfTrips', 'NumberOfChildrenVisiting'
]
categorical_features = [
    'TypeofContact', 'Occupation', 'Gender',
    'ProductPitched', 'MaritalStatus', 'Designation'
]

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Model
xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

param_grid = {
    'xgbregressor__n_estimators': [50, 100],
    'xgbregressor__max_depth': [3, 5],
    'xgbregressor__learning_rate': [0.01, 0.05],
    'xgbregressor__subsample': [0.7, 0.8],
    'xgbregressor__colsample_bytree': [0.7, 0.8],
    'xgbregressor__reg_lambda': [0.1, 1]
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    grid_search = GridSearchCV(
        model_pipeline, param_grid,
        cv=3, n_jobs=-1, scoring='neg_mean_squared_error'
    )
    grid_search.fit(Xtrain, ytrain)

    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("mean_neg_mse", results['mean_test_score'][i])

    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test  = best_model.predict(Xtest)

    mlflow.log_metrics({
        "train_RMSE": mean_squared_error(ytrain, y_pred_train),
        "test_RMSE":  mean_squared_error(ytest,  y_pred_test),
        "train_MAE":  mean_absolute_error(ytrain, y_pred_train),
        "test_MAE":   mean_absolute_error(ytest,  y_pred_test),
        "train_R2":   r2_score(ytrain, y_pred_train),
        "test_R2":    r2_score(ytest,  y_pred_test)
    })

    # Save model locally
    model_path = "tourism_project_test.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved: {model_path}")

# Upload model to HF
model_repo_id   = "Pradeepec59/Tourism"
model_repo_type = "model"

try:
    api.repo_info(repo_id=model_repo_id, repo_type=model_repo_type)
    print(f"Repo '{model_repo_id}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=model_repo_id, repo_type=model_repo_type, private=False)
    print(f"Repo '{model_repo_id}' created.")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="tourism_project_test.joblib",
    repo_id=model_repo_id,
    repo_type=model_repo_type,
)
print("Model uploaded to Hugging Face successfully.")
