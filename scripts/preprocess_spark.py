from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import IntegerType, FloatType, StringType
import os
import yaml

RAW_DIR = "data/raw"
OUT_DIR = "features"
SCHEMA_PATH = "expectations/schemas/titanic.yaml"

def load_schema(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def spark_type_name(spark_type):
    if isinstance(spark_type, IntegerType):
        return "integer"
    elif isinstance(spark_type, FloatType):
        return "float"
    elif isinstance(spark_type, StringType):
        return "string"
    else:
        return str(spark_type)

def validate_schema(df, schema_yaml):
    schema = load_schema(schema_yaml)
    expect_cols = set(schema['columns'].keys())
    actual_cols = set(df.columns)

    # 1. Check for required columns missing
    missing = expect_cols - actual_cols
    if missing:
        raise Exception(f"Missing required columns: {missing}")

    # 2. Check for forbidden extra columns
    if schema.get('forbid_extra_columns', False):
        extras = actual_cols - expect_cols
        if extras:
            raise Exception(f"Unexpected extra columns: {extras}")

    # 3. Check column types
    for col_name, col_rule in schema['columns'].items():
        if col_name not in df.columns:
            continue
        expected_type = col_rule['type']
        actual_type = df.schema[col_name].dataType
        if spark_type_name(actual_type) != expected_type:
            raise Exception(
                f"Column '{col_name}' type mismatch: "
                f"expected {expected_type}, found {spark_type_name(actual_type)}"
            )

spark = SparkSession.builder.appName("TitanicPreprocessing").getOrCreate()
input_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
if not input_files:
    raise Exception("No CSV files found in data/raw/")
input_path = os.path.join(RAW_DIR, input_files[0])
df = spark.read.csv(input_path, header=True, inferSchema=True)

dtype_map = {
    "Survived": IntegerType(),   # Only in train.csv
    "Pclass": IntegerType(),
    "Name": StringType(),
    "Sex": StringType(),
    "Age": FloatType(),
    "SibSp": IntegerType(),
    "Parch": IntegerType(),
    "Ticket": StringType(),
    "Fare": FloatType(),
    "Cabin": StringType(),
    "Embarked": StringType()
}
allowed_cols = list(dtype_map.keys())
df = df.select(*[c for c in allowed_cols if c in df.columns])
for col_name, t in dtype_map.items():
    if col_name in df.columns:
        df = df.withColumn(col_name, col(col_name).cast(t))
# Schema validation
validate_schema(df, SCHEMA_PATH)
# Drop rows with nulls in key columns
required_cols = [c for c in ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"] if c in df.columns]
df = df.dropna(subset=required_cols)
for col_name in ["Sex", "Embarked"]:
    if col_name in df.columns:
        indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_idx", handleInvalid="keep")
        df = indexer.fit(df).transform(df).drop(col_name).withColumnRenamed(col_name + "_idx", col_name)
if "Survived" in df.columns:
    df = df.withColumn("Survived", col("Survived").cast("integer"))

# Drop string/categorical columns that are not useful for modeling
cols_to_drop = ["Name", "Ticket", "Cabin"]
df = df.drop(*[c for c in cols_to_drop if c in df.columns])

df.write.mode("overwrite").parquet(OUT_DIR)
spark.stop()
print("Preprocessing complete. Output saved to features/")

