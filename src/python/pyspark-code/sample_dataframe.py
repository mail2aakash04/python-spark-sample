from pyspark.sql import SparkSession
from pyspark.sql.functions import when
import logging
logger = logging.getLogger(__name__)

spark = SparkSession.builder.appName("SaltingExample").getOrCreate()

data = [
    ("A", 10),
    ("A", 20),
    ("A", 30),
    ("B", 5),
    ("C", 7)
]
df = spark.createDataFrame(data, ["key", "value"])

df.show(truncate=False)