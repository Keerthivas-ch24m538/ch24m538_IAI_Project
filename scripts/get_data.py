import yaml
import requests
import os  # <-- add this

# Load params from params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)
url = params["get_data"]["dataset_url"]
out_path = params["get_data"]["out_path"]

# Ensure the parent directory exists
out_dir = os.path.dirname(out_path)
os.makedirs(out_dir, exist_ok=True)

response = requests.get(url)
response.raise_for_status()  # Fail on error

with open(out_path, "wb") as f:
    f.write(response.content)

print(f"Downloaded dataset from {url} to {out_path}")

