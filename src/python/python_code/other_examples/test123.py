
from pyspark.sql import SparkSession
from pyspark.sql.functions import window, col, row_number, desc
from pyspark.sql.window import Window

events = [
    {"event_id": "e1", "timestamp": "2025-06-24T10:00:00", "user_id": "u1"},
    {"event_id": "e2", "timestamp": "2025-06-24T10:01:00", "user_id": "u2"},
    {"event_id": "e1", "timestamp": "2025-06-24T10:02:00", "user_id": "u3"},
]

spark =  SparkSession.builder.appName("TestApp").getOrCreate()
eventsDF = spark.createDataFrame(events)
eventsDF.show()

windowspec = Window.partitionBy("event_id").orderBy(desc("timestamp"))

eventsDFFinal = eventsDF.withColumn("rank1", row_number().over(windowspec))

eventsDF_ouput1 = eventsDFFinal.filter(col("rank1") == 1).drop("rank1")

output = eventsDF_ouput1.collect()

out = []
for i in output:
    x = i.asDict(i)
    out.append(x)

print(out)









