from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("aakash_app") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("FATAL")

data = [
  "col_1, col_2, col_3",
  "1, ABC, Foo1",
  "2, ABCD, Foo2",
  "3, ABCDE, Foo3",
  "4, ABCDEF, Foo4",
  "5, DEF, Foo5",
  "6, DEFGHI, Foo6",
  "7, GHI, Foo7",
  "8, GHIJKL, Foo8",
  "9, JKLMNO, Foo9",
  "10, MNO, Foo10"
]
rows = [tuple(item for item in line.split(",")) for line in data]

print(rows)
full_csv = spark.sparkContext.parallelize(data)

header_row = full_csv.first()
schema = header_row.split(", ")

data = full_csv.filter(lambda row: row != header_row).map(lambda x: x.split(","))

for x in data.collect():
    print(x)

df = spark.createDataFrame(data, schema=schema)
df.show(truncate=False)