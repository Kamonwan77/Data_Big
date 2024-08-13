from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType

# สร้าง SparkSession
spark = SparkSession.builder \
    .appName("StreamingDataFrames") \
    .config("spark.master", "local") \
    .getOrCreate()

# กำหนด Schema สำหรับข้อมูลในไฟล์
file_schema = StructType([
    StructField("status_id", StringType(), True),
    StructField("status_type", StringType(), True),
    StructField("status_published", StringType(), True),
    StructField("num_reactions", StringType(), True),
    StructField("num_comments", StringType(), True),
    StructField("num_shares", StringType(), True),
    StructField("num_likes", StringType(), True),
    StructField("num_loves", StringType(), True),
    StructField("num_wows", StringType(), True),
    StructField("num_hahas", StringType(), True),
    StructField("num_sads", StringType(), True),
    StructField("num_angrys", StringType(), True)
])

# อ่านข้อมูลจาก File (โดยใช้การอ่านไฟล์ที่เพิ่มขึ้น พร้อมกับการใช้ Schema ที่กำหนด)
lines = spark.readStream \
    .format("csv") \
    .option("header", "true") \
    .option("maxFilesPerTrigger", 1) \
    .option("path", "C:/Users/gamon/Documents/Data/Data_Big/stream") \
    .schema(file_schema) \
    .load()

# พิมพ์ schema ของ DataFrame
lines.printSchema()

# การจัดการคอลัมน์และการตั้งค่า watermark
words = lines.withColumn("date", split(col("status_published"), " ").getItem(1)) \
    .withColumn("timestamp", current_timestamp()) \
    .withWatermark("timestamp", "10 seconds")

# นับจำนวนคำ
wordCounts = words.groupBy("date", "status_type", "timestamp").count()

# เขียนผลลัพธ์ไปยังไฟล์ CSV
query = wordCounts.writeStream \
    .format("csv") \
    .option("path", "C:/Users/gamon/Documents/Data/Data_Big/savetofile") \
    .trigger(processingTime='5 seconds') \
    .option("checkpointLocation", "C:/Users/gamon/Documents/Data/Data_Big/savetofile/checkpoint") \
    .outputMode("append") \
    .option("truncate", "false") \
    .start()

# รอให้กระบวนการ streaming ทำงาน
query.awaitTermination()

# อ่านข้อมูลจาก Socket
socket_stream_df = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load() \
    .selectExpr("value as message")

# อ่านข้อมูลจาก Kafka
kafka_stream_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "topic_name") \
    .load() \
    .selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")

# การเขียนผลลัพธ์จาก Socket และ Kafka ไปยัง Console
query_socket = socket_stream_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query_kafka = kafka_stream_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

# รอให้กระบวนการ streaming ทำงาน
query_socket.awaitTermination()
query_kafka.awaitTermination()
