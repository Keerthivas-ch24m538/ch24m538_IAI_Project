import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os

# Set MLflow tracking to your running server/UI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("default")  # Change name if you want a new experiment

# Find and read all Parquet feature files
features_dir = "features"
all_parquet = [
    os.path.join(features_dir, f)
    for f in os.listdir(features_dir)
    if f.endswith(".parquet")
]
df = pd.concat([pd.read_parquet(fp) for fp in all_parquet], ignore_index=True)

# Prepare data for training
target_col = "Survived"
X = df.drop([target_col], axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    clf = LogisticRegression(max_iter=100)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(clf, "model")
    print(f"Logged MLflow run: Accuracy={acc:.3f}")

