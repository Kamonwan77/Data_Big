from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql.functions import desc, col, lit

# สร้าง SparkSession
spark = SparkSession.builder \
    .appName("CyclingRoutesGraph") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .getOrCreate()

# อ่านข้อมูลจากไฟล์ cycling.csv
cycling_routes_df = spark.read.csv("C:\\Users\\ADMIN\\Documents\\Data_Big\\Quiz2-TEST\\GraphAnal Cycli\\cycling.csv", header=True, inferSchema=True)

# แสดง DataFrame
cycling_routes_df.show()

# สร้าง DataFrame สำหรับ vertices โดยใช้ withColumnRenamed() และ FromStationName เป็น id
vertices = cycling_routes_df.select("FromStationName").withColumnRenamed("FromStationName", "id").distinct()

# สร้าง DataFrame สำหรับ edges โดยใช้ withColumnRenamed() โดยให้ FromStationName เป็น src และ ToStationName เป็น dst
edges = cycling_routes_df.select("FromStationName", "ToStationName") \
    .withColumnRenamed("FromStationName", "src") \
    .withColumnRenamed("ToStationName", "dst")

# แสดง DataFrame สำหรับ vertices
vertices.show()

# แสดง DataFrame สำหรับ edges
edges.show()

# สร้าง GraphFrame โดยใช้ vertices และ edges ที่สร้างขึ้น
graph = GraphFrame(vertices, edges)

# แสดงจำนวน vertices
print("Number of vertices:", graph.vertices.count())

# แสดงจำนวน edges
print("Number of edges:", graph.edges.count())

# กลุ่ม edges โดยใช้ src และ dst, กรองตาม count > 5, เพิ่มคอลัมน์ source_color และ destination_color
grouped_edges = graph.edges.groupBy("src", "dst").count() \
    .filter(col("count") > 5) \
    .withColumn("source_color", lit("#FF3F33")) \
    .withColumn("destination_color", lit("#3358FF")) \


# กลุ่ม edges โดยใช้ src และ dst, กรองตาม count > 5, เพิ่มคอลัมน์ source_color และ destination_color
grouped_edges1 = graph.edges.groupBy("src", "dst").count() \
    .filter(col("count") > 5) \
    .withColumn("source_color", lit("#FF3F33")) \
    .withColumn("destination_color", lit("#3358FF")) \
    .drop("count")

# แสดงข้อมูลที่ถูกจัดกลุ่ม
grouped_edges.show()

# เขียนข้อมูลที่ถูกจัดกลุ่มลงในไฟล์ CSV โดยใช้โหมด overwrite และตั้ง header เป็น True
grouped_edges1.write.csv("grouped_cycling_routes.csv", mode="overwrite" , header="false")

print('='*80)
print('')
