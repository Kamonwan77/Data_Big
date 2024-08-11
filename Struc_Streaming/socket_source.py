from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split

# Create a Spark session
spark = SparkSession.builder.getOrCreate()

# Create a streaming DataFrame that reads from a socket
lines = spark.readStream.format("socket")\
    .option("host", "localhost")\
    .option("port", 9999)\
    .load()

# Split the lines into words
words = lines.select(
    explode(
        split(lines.value, " ")
    ).alias("word")
)

# Group the words and count the occurrences
wordCounts = words.groupBy("word").count()

# Start the query to write the results to the console
query = wordCounts.writeStream\
    .outputMode("complete")\
    .format("console")\
    .start()

# Wait for the query to terminate
query.awaitTermination()