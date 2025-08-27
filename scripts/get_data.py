import yaml
import requests

# Load params from params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)

url = params["get_data"]["dataset_url"]
out_path = params["get_data"]["out_path"]

response = requests.get(url)
response.raise_for_status()  # Fail on error

with open(out_path, "wb") as f:
    f.write(response.content)

print(f"Downloaded dataset from {url} to {out_path}")

