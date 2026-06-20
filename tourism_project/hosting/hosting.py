from huggingface_hub import HfApi
import os

# ✅ Works in both Colab and GitHub Actions
try:
    from google.colab import userdata
    hf_token = userdata.get("HF_TOKEN")
except ImportError:
    hf_token = os.environ.get("HF_TOKEN")

if hf_token is None:
    raise ValueError("HF_TOKEN not found in Colab Secrets or GitHub Actions Secrets.")

api = HfApi(token=hf_token)

# ✅ Use the correct local path (relative to where the script runs)
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id="Pradeepec59/Tourism",
    repo_type="space",
    path_in_repo="",
)

print("Files uploaded to Hugging Face Space successfully.")
