from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, current_timestamp, window
from pyspark.sql import functions as F

# สร้าง SparkSession
spark = SparkSession.builder \
    .appName("StreamingWindowCounts") \
    .config("spark.master", "local") \
    .getOrCreate()

# อ่านข้อมูลจาก Socket
lines = spark.readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 9999) \
    .load() \
    .selectExpr("value as message")

# แยกคำและเพิ่มคอลัมน์ timestamp
words = lines.withColumn("date", split(col("message"), " ").getItem(1)) \
    .withColumn("timestamp", F.current_timestamp()) \
    .withWatermark("timestamp", "10 seconds") \
    .select("date", "timestamp", split(col("message"), " ").alias("words"))

# ใช้ window function เพื่อทำการนับจำนวนคำในช่วงเวลาที่กำหนด
windowCounts = words.select(
    F.explode(col("words")).alias("word"),
    col("timestamp")
).groupBy(
    window(col("timestamp"), "10 seconds", "5 seconds"),
    col("word")
).count()

# เขียนผลลัพธ์ไปยัง Console
query = windowCounts.writeStream \
    .outputMode("update") \
    .format("console") \
    .option("truncate", "false") \
    .start()

# รอให้กระบวนการ streaming ทำงาน
query.awaitTermination()
