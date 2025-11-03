from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("aakash_app") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("FATAL")

data = [
"Row-Key-001,K1,10,A2,20,K3,30,B4,42,K5,19,C20,20",
"Row-Key-002,X1,20,Y6,10,Z15,35,X16,42",
"Row-Key-003,L4,30,M10,5,N12,38,O14,41,P13,8"
]

rows = [tuple(item.strip() for item in line.split(",")) for line in data]

full_csv = spark.sparkContext.parallelize(data)

output = full_csv.map(lambda x: x.split(","))

for x in output.collect():
    print(x)
