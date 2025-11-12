from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("aakash_app") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("FATAL")

input_data = [
    "Row-Key-001, K1, 10, A2, 20, K3, 30, B4, 42, K5, 19, C20, 20",
    "Row-Key-002, X1, 20, Y6, 10, Z15, 35, X16, 42",
    "Row-Key-003, L4, 30, M10, 5, N12, 38, O14, 41, P13, 8"
]

# Create RDD
data_lines = spark.sparkContext.parallelize(input_data)

# Transform the data
key_values = data_lines.flatMap(lambda line: create_output_format(line))

def create_output_format(line):
    values = [x.strip() for x in line.split(",")]
    startingKey = values[0]
    keys = values[1::2]  # Take every other starting from index 1
    return [(startingKey, k) for k in keys]

# Collect and print
for y in key_values.collect():
    print(y)