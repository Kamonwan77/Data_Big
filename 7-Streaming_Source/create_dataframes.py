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

# อ่านข้อมูลจากไฟล์ (ใช้การอ่านไฟล์ที่เพิ่มขึ้น พร้อมกับการใช้ Schema ที่กำหนด)
lines = spark.readStream \
    .format("csv") \
    .option("header", "true") \
    .option("maxFilesPerTrigger", 1) \
    .option("path", "C:\\Users\\gamon\\Documents\\Data\\Data_Big\\7-Streaming_Source\stream") \
    .schema(file_schema) \
    .load()

# พิมพ์ schema ของ DataFrame เพื่อยืนยันการอ่านข้อมูล
lines.printSchema()

# แสดงข้อมูลบางส่วนจาก DataFrame เพื่อยืนยันการอ่านข้อมูล
lines.writeStream \
    .format("console") \
    .start() \
    .awaitTermination(10)  # รอการพิมพ์ข้อมูล 10 วินาที แล้วหยุดการทำงาน

# การจัดการคอลัมน์และการตั้งค่า watermark
words = lines.withColumn("date", split(col("status_published"), " ").getItem(1)) \
    .withColumn("timestamp", current_timestamp()) \
    .withWatermark("timestamp", "10 seconds")

# นับจำนวนคำ
wordCounts = words.groupBy("date", "status_type", "timestamp").count()

# เขียนผลลัพธ์ไปยังไฟล์ CSV (ใช้ append mode แทน complete mode)
query = wordCounts.writeStream \
    .format("csv") \
    .option("path", "C:\\Users\\gamon\\Documents\\Data\\Data_Big\\7-Streaming_Source\\savetofile\\checkpoint") \
    .trigger(processingTime='5 seconds') \
    .option("checkpointLocation", "C:\\Users\\gamon\\Documents\\Data\\Data_Big\\7-Streaming_Source\\savetofile\\checkpoint") \
    .outputMode("append") \
    .option("truncate", "false") \
    .start()

# รอให้กระบวนการ streaming ทำงาน
query.awaitTermination()