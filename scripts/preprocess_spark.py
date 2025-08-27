from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer
import os

RAW_DIR = "data/raw"
OUT_DIR = "features"

spark = SparkSession.builder.appName("TitanicPreprocessing").getOrCreate()

# Find the Titanic CSV file in raw/
input_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
if not input_files:
    raise Exception("No CSV files found in data/raw/")
input_path = os.path.join(RAW_DIR, input_files[0])

# Read CSV
df = spark.read.csv(input_path, header=True, inferSchema=True)

# Drop unnecessary columns (optional—uncomment as needed)
# cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
# df = df.drop(*cols_to_drop)

# Drop rows with nulls in key columns
required_cols = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
df = df.dropna(subset=required_cols)

# Encode 'Sex' and 'Embarked'
for col_name in ["Sex", "Embarked"]:
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_idx", handleInvalid="keep")
    df = indexer.fit(df).transform(df).drop(col_name).withColumnRenamed(col_name + "_idx", col_name)

# Ensure Survived is integer
df = df.withColumn("Survived", col("Survived").cast("integer"))

# Write to features/ as Parquet (overwrite if exists)
df.write.mode("overwrite").parquet(OUT_DIR)

spark.stop()
print("Preprocessing complete. Output saved to features/")

