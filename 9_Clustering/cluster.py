# Import libraries
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

import matplotlib.pyplot as plt
import pandas as pd

# สร้าง SparkSession
spark = SparkSession \
    .builder \
    .appName("testKMeans") \
    .getOrCreate()

# โหลดไฟล์ CSV โดยมี header
df = spark.read.format("csv") \
    .option("header", True) \
    .option("inferSchema", True) \
    .load('C:\\Users\\gamon\\Documents\\Data\\Data_Big\\9-Clustering\\fb_live_thailand.csv')

# แปลงข้อมูลเป็น Double
df = df.select(df.num_sads.cast(DoubleType()),
               df.num_reactions.cast(DoubleType()))

# รวมคอลัมน์เพื่อสร้างฟีเจอร์ "features"
vec_assembler = VectorAssembler(inputCols = ["num_sads",
                                             "num_reactions"],
                                outputCol = "features")

# ทำการ Scaling เพื่อให้คอลัมน์มีค่าที่เปรียบเทียบกันได้
scaler = StandardScaler(inputCol = "features",
                        outputCol = "scaledFeatures",
                        withStd = True,
                        withMean = False)

# สร้าง list เพื่อเก็บค่า k ที่ดีที่สุด
k_values = []

# วนลูปเพื่อหาค่า k ที่เหมาะสมในช่วง 2 ถึง 5
for k in range(2, 5):
    kmeans = KMeans(featuresCol = "scaledFeatures",
                    predictionCol = "prediction_col",
                    k = k)
    pipeline = Pipeline(stages = [vec_assembler, scaler, kmeans])
    model = pipeline.fit(df)
    output = model.transform(df)
    evaluator = ClusteringEvaluator(predictionCol = "prediction_col",
                                    featuresCol = "scaledFeatures",
                                    metricName = "silhouette",
                                    distanceMeasure = "squaredEuclidean")
    score = evaluator.evaluate(output)
    k_values.append(score)
    print(f"Silhouette score for k={k}: {score}")

# หาค่า k ที่ดีที่สุด
best_k = k_values.index(max(k_values)) + 2
print(f"The best k: {best_k} with silhouette score: {max(k_values)}")

# เริ่มต้นการทำงานของ KMeans ด้วยค่า k ที่ดีที่สุด
kmeans = KMeans(featuresCol = "scaledFeatures",
                predictionCol = "prediction_col",
                k = best_k)

# สร้าง pipeline
pipeline = Pipeline(stages = [vec_assembler, scaler, kmeans])

# Fit model
model = pipeline.fit(df)

# ทำการพยากรณ์ผลลัพธ์
predictions = model.transform(df)

# ประเมินผลลัพธ์
evaluator = ClusteringEvaluator(predictionCol = "prediction_col",
                                featuresCol = "scaledFeatures",
                                metricName = "silhouette",
                                distanceMeasure = "squaredEuclidean")

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# แปลงผลลัพธ์เป็น Pandas DataFrame เพื่อการ visualizing
clustered_data_pd = predictions.toPandas()

# Visualization ของผลลัพธ์
plt.scatter(clustered_data_pd["num_reactions"],
            clustered_data_pd["num_sads"],
            c = clustered_data_pd["prediction_col"])
plt.xlabel("num_reactions")
plt.ylabel("num_sads")
plt.title("k-means Clustering")
plt.colorbar().set_label("Cluster")
plt.show()