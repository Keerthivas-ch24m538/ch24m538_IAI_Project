import pandas as pd
import sys

required_columns = ["Pclass","Name", "Sex", "Fare", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"]
sex_values = {"male", "female"}
embarked_values = {"C", "Q", "S", None, pd.NA}
pclass_values = {1, 2, 3}

df = pd.read_csv("data/raw/titanic.csv")  # use the real file path

# Columns check
missing = set(required_columns) - set(df.columns)
extra = set(df.columns) - set(required_columns + ["PassengerId", "Survived"])
if missing:
    sys.exit(f"Missing columns: {missing}")
if extra:
    sys.exit(f"Extra columns: {extra}")

# Null checks
for col in ["Pclass", "Sex", "Fare"]:
    if df[col].isnull().any():
        sys.exit(f"Nulls in {col}")
if df["Age"].isnull().any():
    print("Warning: Nulls present in Age (allowed)")

# Value checks
if not set(df["Sex"].unique()).issubset(sex_values):
    sys.exit("Unexpected values in Sex")
if not set(df["Embarked"].dropna().unique()).issubset(embarked_values):
    sys.exit("Unexpected values in Embarked")
if not set(df["Pclass"].unique()).issubset(pclass_values):
    sys.exit("Unexpected values in Pclass")
if (df["Fare"] < 0).any():
    sys.exit("Some Fare values are negative")

if "Survived" in df:
    if not set(df["Survived"].dropna().unique()).issubset({0,1}):
        sys.exit("Unexpected values in Survived")

# Deduplicate by PassengerId if present
if "PassengerId" in df:
    before = len(df)
    df = df.drop_duplicates(subset="PassengerId")
    after = len(df)
    if before != after:
        print(f"Removed {before-after} duplicate rows based on PassengerId")

print("Raw Titanic data validation passed.")

