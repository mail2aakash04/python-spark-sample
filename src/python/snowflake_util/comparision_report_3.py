from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
import json
import ollama

# =====================================================
# STEP 1: Initialize Spark
# =====================================================
spark = SparkSession.builder.appName("Readiness and Schema Analysis").getOrCreate()

# =====================================================
# STEP 2: Helper to lowercase all string columns
# =====================================================
def lowercase_all_string_cols(df):
    for col_name, dtype in df.dtypes:
        if dtype == "string":
            df = df.withColumn(col_name, F.lower(F.col(col_name)))
    return df

# =====================================================
# STEP 3: Read both CSVs
# =====================================================
sf_df = spark.read.option("header", True).csv("./snowflake_tables.csv")
sole_df = spark.read.option("header", True).csv("./sole_tables.csv")

# Normalize column names
sf_df = sf_df.select([F.col(c).alias(c.strip().lower()) for c in sf_df.columns])
sole_df = sole_df.select([F.col(c).alias(c.strip().lower()) for c in sole_df.columns])

# Convert string columns to lowercase
sf_df = lowercase_all_string_cols(sf_df)
sole_df = lowercase_all_string_cols(sole_df)

# =====================================================
# STEP 4: Coverage Analysis Report
# =====================================================
sf_tables = sf_df.select("table_name").distinct()
sole_tables = sole_df.select("table_name").distinct()

matched = sf_tables.join(sole_tables, "table_name", "inner")
missing = sf_tables.join(sole_tables, "table_name", "left_anti")

total_sf = sf_tables.count()
matched_count = matched.count()
missing_count = missing.count()
coverage_rate = round((matched_count / total_sf) * 100, 2) if total_sf > 0 else 0.0

print("\n===== Coverage Analysis Report =====\n")
print(f"Total Snowflake Objects: {total_sf}")
print(f"Objects Matched with SOLE: {matched_count}")
print(f"Coverage Rate: {coverage_rate}%")

# =====================================================
# STEP 5: Schema Analysis with Type Normalization
# =====================================================

def normalize_dtype(dtype):
    if dtype is None:
        return None
    dtype = dtype.lower().strip()
    mapping = {
        "timestamp_ntz": "timestamp",
        "timestamp_ltz": "timestamp",
        "timestamp_tz": "timestamp",
        "text": "string",
        "varchar": "string",
        "char": "string",
        "number": "decimal",
        "int": "decimal",
        "integer": "decimal",
        "bigint": "decimal",
        "float": "decimal",
        "double": "decimal",
    }
    return mapping.get(dtype, dtype)

normalize_dtype_udf = F.udf(normalize_dtype, T.StringType())

sf_df = sf_df.withColumn("normalized_data_type", normalize_dtype_udf(F.col("data_type")))
sole_df = sole_df.withColumn("normalized_data_type", normalize_dtype_udf(F.col("data_type")))

matched_tables = matched.select("table_name")

schema_comparisons = []

for row in matched_tables.collect():
    tbl = row["table_name"]

    sf_schema = sf_df.filter(F.col("table_name") == tbl) \
                     .select(F.col("column_name").alias("col_name"),
                             F.col("normalized_data_type").alias("sf_type"))
    sole_schema = sole_df.filter(F.col("table_name") == tbl) \
                         .select(F.col("column_name").alias("col_name"),
                                 F.col("normalized_data_type").alias("sole_type"))

    merged = sf_schema.join(sole_schema, "col_name", "outer") \
        .withColumn("match",
                    F.when(
                        (F.col("sf_type").isNotNull()) &
                        (F.col("sole_type").isNotNull()) &
                        (F.col("sf_type") == F.col("sole_type")),
                        1
                    ).otherwise(0))

    total_cols = merged.count()
    if total_cols == 0:
        continue

    matched_cols = merged.filter(F.col("match") == 1).count()
    schema_match_pct = round((matched_cols / total_cols) * 100, 2)

    schema_comparisons.append((tbl, matched_cols, total_cols, schema_match_pct))

schema_df = spark.createDataFrame(schema_comparisons, ["table_name", "matched_cols", "total_cols", "schema_match_pct"])

matched_objects = schema_df.count()
exact_schema = schema_df.filter(F.col("schema_match_pct") == 100).count()
mismatch_schema = matched_objects - exact_schema
match_percentage = round((exact_schema / matched_objects) * 100, 2) if matched_objects else 0

print("\n===== Schema Analysis Report =====\n")
print(f"Matched Objects: {matched_objects}")
print(f"Objects with Exactly Same Schema: {exact_schema}")
print(f"Objects with Mismatched Schema: {mismatch_schema}")
print(f"Match Percentage (%): {match_percentage}")

# =====================================================
# STEP 6: Example Mismatched Table with Full Schema Diff
# =====================================================
mismatched_table_row = schema_df.filter(F.col("schema_match_pct") < 100).limit(1).collect()

if mismatched_table_row:
    mismatched_table = mismatched_table_row[0]["table_name"]
    print(f"\n🔍 Example Mismatched Table: {mismatched_table}\n")

    sf_schema = sf_df.filter(F.col("table_name") == mismatched_table) \
                     .select(F.col("column_name").alias("col_name"),
                             F.col("normalized_data_type").alias("sf_type"))
    sole_schema = sole_df.filter(F.col("table_name") == mismatched_table) \
                         .select(F.col("column_name").alias("col_name"),
                                 F.col("normalized_data_type").alias("sole_type"))

    # ✅ FULL OUTER JOIN — covers extra or missing columns in either
    schema_diff = sf_schema.join(sole_schema, "col_name", "outer") \
        .withColumn(
            "match_status",
            F.when(
                (F.col("sf_type").isNull()) & (F.col("sole_type").isNotNull()),
                "ONLY_IN_SOLE"
            ).when(
                (F.col("sf_type").isNotNull()) & (F.col("sole_type").isNull()),
                "ONLY_IN_SNOWFLAKE"
            ).when(
                (F.col("sf_type") == F.col("sole_type")),
                "MATCH"
            ).otherwise("MISMATCH")
        )

    print(f"=== Detailed Schema Difference for '{mismatched_table}' ===")
    schema_diff.groupBy("match_status").count().show()
else:
    print("\n✅ All matched tables have identical schemas.")

readiness_score = (coverage_rate * 0.4) + (match_percentage * 0.6)
print(f"✅ Readiness Score : {round(readiness_score, 2)}%")


# =====================================================
# STEP 7: Optional - Generate Readiness Summary using LLaMA
# =====================================================
report_data = {
    "coverage_analysis": {
        "total_snowflake_objects": total_sf,
        "objects_matched_in_sole": matched_count,
        "coverage_rate_percent": coverage_rate,
    },
    "schema_analysis": {
        "matched_objects": matched_objects,
        "exact_schema_match": exact_schema,
        "mismatched_schema": mismatch_schema,
        "match_percentage": match_percentage
    },
   "readiness_score": {
         "readiness_score" : readiness_score
   }
}

report_json = json.dumps(report_data, indent=2)
prompt = f"""
Generate a clear and professional readiness report.
Include:
- A summary paragraph
- Each metric in bullet points
- A final conclusion about coverage completeness.

Can you compare the schema of matched tables and let me know how much percentage of columns are matching vs mismatching?

"""

response = ollama.chat(model="llama3.2:3b", messages=[
    {"role": "user", "content": prompt}
])

print("\n===== LLaMA Generated Readiness Summary =====\n")
print(response['message']['content'])
